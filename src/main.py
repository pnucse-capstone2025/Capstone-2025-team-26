import asyncio
import os
import difflib
import subprocess
import time
import json
import clang.cindex
import shutil
import sys
import re
import sqlite3

import parser.c_parser
import limiter.sliding_window

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import stdio_client
from datetime import datetime
from pathlib import Path
from typing import List, Type

# .env 로드 및 환경변수
load_dotenv()
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY       = os.getenv("ANTHROPIC_API_KEY")

MODEL_NAME              = os.getenv("MODEL_NAME")

SONARQUBE_TOKEN         = os.getenv("SONARQUBE_TOKEN")
SONARQUBE_ORG           = os.getenv("SONARQUBE_ORG")
SONARQUBE_PROJECT_KEY   = os.getenv("SONARQUBE_PROJECT_KEY")

PROJECT_ROOT            = os.getenv("PROJECT_ROOT")

GITHUB_PAT_KEY          = os.getenv("GITHUB_PAT_KEY") 
GITHUB_USER_NAME        = os.getenv("GITHUB_USER_NAME") 
GITHUB_REPO_NAME        = os.getenv("GITHUB_REPO_NAME")
GIT_WORKDIR             = os.getenv("GIT_WORKDIR", PROJECT_ROOT)
TARGET_BRANCH           = os.getenv("TARGET_BRANCH", "main")

WAIT_SECONDS            = int(os.getenv("WAIT_SECONDS", "75"))
MAX_ROUNDS              = int(os.getenv("MAX_ROUNDS", "10"))

DB_PATH = Path(os.getenv("DB_PATH") + "\\agent_activity.db")

from pydantic import BaseModel, Field
class Patch(BaseModel):
    code: str = Field(..., description="entire code")
    no_change: bool = Field(..., description="변경 사항이 있다면 True")
    has_more: bool = Field(..., description="추가적인 부분이 있다면 True")
    part_index: int = Field(..., description="0부터 시작하는 조각 인덱스")

class CommitSubject(BaseModel):
    subject: str = Field(..., description="Single-line English commit subject")

def build_llm(model_name: str,
              patch_model: Type[BaseModel],
              *,
              temperature: float = 0.0):
    m = model_name.lower().strip()
    # 1) Gemini 계열
    if m.startswith("gemini"):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required for Gemini models")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            response_schema=patch_model.model_json_schema(),
            response_mime_type="application/json",
        ).with_structured_output(patch_model)

    # 2) OpenAI GPT 계열
    if m.startswith("gpt-"):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI API key required for OpenAI models")
        # 가장 호환성 높은 JSON 모드 사용
        return ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        ).with_structured_output(patch_model, method="json_mode")

    # 3) Anthropic Claude 계열
    if m.startswith("claude"):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC API KEY required for Anthropic models")
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(patch_model)

    raise ValueError(f"Unknown or unsupported model name: {model_name}")

llm = build_llm(
    MODEL_NAME,
    Patch,
    temperature=0.0,
)

commit_subject_llm  = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    response_schema=CommitSubject.model_json_schema(),
    response_mime_type="application/json",
).with_structured_output(CommitSubject)

def _init():
    """필수 환경 변수 체크"""
    required_envs = [
        ("SONARQUBE_TOKEN", os.getenv("SONARQUBE_TOKEN")),
        ("SONARQUBE_ORG", os.getenv("SONARQUBE_ORG")),
        ("SONARQUBE_PROJECT_KEY", os.getenv("SONARQUBE_PROJECT_KEY")),
        ("PROJECT_ROOT", os.getenv("PROJECT_ROOT")),
        ("GITHUB_PAT_KEY", os.getenv("GITHUB_PAT_KEY")),
        ("LIBCLANG_PATH", os.getenv("LIBCLANG_PATH"))
    ]
    missing = [name for name, value in required_envs if not value or value.strip() == ""]
    if missing:
        print("\n[ERROR] 필수 환경변수 누락 또는 값 없음:")
        for name in missing:
            print(f"  - {name}")
        print("\n[INFO] .env 파일과 실행 환경을 확인해 주세요.")
        sys.exit(1)
    # clang 세팅
    LIBCLANG_PATH = os.getenv("LIBCLANG_PATH")
    if LIBCLANG_PATH and os.path.exists(LIBCLANG_PATH):
        clang.cindex.Config.set_library_file(LIBCLANG_PATH)

RE_ENTER  = re.compile(r"^(?:g?make)\[\d+\]: Entering directory ['`]([^'`]+)['`]")
RE_ERR = re.compile(
    r"^\s*(?P<file>(?:[A-Za-z]:)?[^:\n]+?\.(?:c|h|cc|cpp|cxx|hpp|hxx))"
    r":(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<level>error|warning|fatal error)\b"
)
RE_INC_1 = re.compile(r"^\s*In file included from\s+([^:]+):(\d+)(?::\d+)?,")
RE_INC_N = re.compile(r"^\s*from\s+([^:]+):(\d+)(?::\d+)?,?$")

def wsl_to_windows(p: str) -> str:
    """ WSL 경로를 Windows 경로로 변환(가능하면 wslpath 사용)"""
    if not p.startswith("/mnt/"):
        return p
    wsl = shutil.which("wsl")
    if wsl:
        r = subprocess.run(["wsl", "wslpath", "-w", p], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    parts = p.split("/")
    if len(parts) > 3 and parts[2]:
        drive = parts[2].upper() + ":"
        rest  = parts[3:]
        return os.path.abspath(os.path.join(drive, *rest))
    return p

def is_project_file(abs_path: str, project_root: str) -> bool:
    try:
        pr = os.path.abspath(project_root)
        ap = os.path.abspath(abs_path)
        return os.path.commonpath([pr, ap]) == pr
    except Exception:
        return False

def resolve_path(tok: str, current_dir: str, project_root: str) -> str:
    # 절대경로(/mnt 포함)는 그대로 정규화
    if tok.startswith("/mnt/"):
        tok = wsl_to_windows(tok)
    # 2) 절대경로면 그대로 정규화
    if os.path.isabs(tok):
        return os.path.abspath(tok)  # Windows 규칙으로 정규화 [web:1366]

    # 3) 상대경로는 현재 빌드/엔터 디렉터리 기준
    base = current_dir or os.path.join(project_root, "build")  # CMake 빌드 트리 기준 [web:1333]
    if base.startswith("/mnt/"):
        base = wsl_to_windows(base)  # 안전 변환 [web:1366]
    return os.path.abspath(os.path.join(base, tok))

def parse_error_paths(log: str, project_root: str) -> set[str]:
    paths = set()
    cur_dir = project_root
    include_stack = []

    for line in log.splitlines():
        m_ent = RE_ENTER.match(line)
        if m_ent:
            cur_dir = m_ent.group(1)
            continue

        m_inc1 = RE_INC_1.match(line)
        if m_inc1:
            include_stack = [m_inc1.group(1)]
            continue
        m_incn = RE_INC_N.match(line)
        if m_incn and include_stack:
            include_stack.append(m_incn.group(1))
            continue

        m_err = RE_ERR.match(line)
        if m_err:
            src = m_err.group(1)
            # 에러 본문 파일
            abs_src = resolve_path(src, cur_dir, project_root)
            if is_project_file(abs_src, project_root):
                paths.add(abs_src)
            # 포함 스택의 프로젝트 내부 헤더들도 보조 후보
            for inc in include_stack:
                abs_inc = resolve_path(inc, cur_dir, project_root)
                if is_project_file(abs_inc, project_root):
                    paths.add(abs_inc)
            include_stack = []
    return paths

def find_header_in_project(rel_path, project_root):
    """프로젝트 전체에서 include 경로(rel_path)와 파일명이 같은 .h 파일을 찾음"""
    header_full_name = os.path.basename(rel_path.strip())
    found = []
    for dirpath, _, filenames in os.walk(project_root):
        for fname in filenames:
            if fname == header_full_name:
                found_path = os.path.abspath(os.path.join(dirpath, fname))
                found.append(found_path)
    return found

def get_related_header_files(c_path, project_root):
    """특정 .c 파일에서 include된 .h 파일을 모두 프로젝트 내에서 찾아 반환"""
    header_files = set()
    with open(c_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'#\s*include\s*"([^"]+\.h)"', line)
            if m:
                rel_path = m.group(1)
                found = find_header_in_project(rel_path, project_root)
                header_files.update(found)
    return header_files

def extract_signatures(project_root, target_files):
    """C 프로젝트 전체 시그니처 요약"""
    c_parser = parser.c_parser.CParser()
    summary = []
    for fpath in target_files:
        try:
            context = c_parser.parse(fpath, project_root)
            summary.append({
                "file": os.path.relpath(fpath, project_root),
                "context": context
            })
        except Exception as e:
            print(f"Parse failed: {fpath} -> {e}")
    return summary

def make_refactor_from_issues(code: str, issues: str, file_path: str, project_context_json: str = "") -> str:   
    prompt_refactor = PromptTemplate(
        input_variables=["code", "issues", "context", "filepath"],
        template=("""
            You are an expert C developer.\n
            Refactor the code to fix the listed SonarCloud issues without changing behavior.\n\n
            Project context (related signatures/types):\n
            <context>\n
            {context}\n
            </context>\n\n
            File: {filepath}\n
            Original C code:\n
            ```c{code}```\n\n
            
            Issues to fix (file-scoped):\n
            <issues>\n
            {issues}\n
            </issues>\n\n

            Output rules:
            - Return only a json object with fields:
                - "code": string, REQUIRED, entire C source after your edits (or original if no_change=true).
                - "no_change": boolean, REQUIRED.    
                - "has_more": boolean, REQUIRED.
            - 변경사항이 없다면 빈 문자열("")을 출력하세요.
            - 한 호출당 출력 길이를 최대한으로 사용하고, 남은 부분이 있으면 has_more=true로 표시해.
                part_index={part_index}에 해당하는 다음 부분만 보내.
            """
        )
    )
    chain = prompt_refactor | llm

    parts: List[str] = []
    part_index = 0
    while part_index < 10:
        print(f"\n==={part_index}===")
        obj: Patch = chain.invoke({
            "code": code,
            "issues": issues,
            "context": project_context_json,
            "filepath": file_path,
            "part_index": part_index
            })
        if obj.no_change and not obj.code.strip():
            break
        if obj.code:
            parts.append(obj.code)
        if not obj.has_more:
            break
        
        part_index += 1

    after_code = "".join(parts)
    markers = ["//", "/*", "#include", "#define", "#if", "#pragma"]
    rst = strip_to_marker(after_code, markers)
    return rst

def make_refactor_from_build(code: str, log: str, file_path: str, project_context_json: str = "") -> str:   
    prompt_refactor = PromptTemplate(
        input_variables=["code", "log", "context", "filepath"],
        template=("""
            You are an expert C developer.\n
            Your mission is to correct the provided C code so that it compiles successfully. \n
            Analyze the build errors from the `make` log to identify the problems, and use the project context to apply the correct fixes. \n
            The goal is to resolve the errors while preserving the original intended functionality.\n
            
            Project context (related signatures/types):\n
            <context>\n
            {context}\n
            </context>\n\n
            
            File: {filepath}\n
            Original C code:\n 
            ```c{code}```\n\n
            
            Build Errors (make log):\n
            <log>\n
            {log}\n
            </log>\n\n


            Output rules:
            - Return only a json object with fields:
                - "code": string, REQUIRED, entire C source after your edits (or original if no_change=true).
                - "no_change": boolean, REQUIRED.    
                - "has_more": boolean, REQUIRED.
            - 변경사항이 없다면 빈 문자열("")을 출력하세요.
            - 한 호출당 출력 길이를 최대한으로 사용하고, 남은 부분이 있으면 has_more=true로 표시해.
                part_index={part_index}에 해당하는 다음 부분만 보내.
            """
        )
    )
    chain = prompt_refactor | llm

    parts: List[str] = []
    part_index = 0
    while part_index < 10:
        print(f"\n==={part_index}===")
        obj: Patch = chain.invoke({
            "code": code,
            "log": log,
            "context": project_context_json,
            "filepath": file_path,
            "part_index": part_index
            })
        if obj.no_change and not obj.code.strip():
            break
        if obj.code:
            parts.append(obj.code)
        if not obj.has_more:
            break
        
        part_index += 1

    after_code = "".join(parts)
    markers = ["//", "/*", "#include", "#define", "#if", "#pragma"]
    rst = strip_to_marker(after_code, markers)
    return rst


def diff_and_maybe_write(path: str, before: str, after: str) -> tuple[bool, str]:
    """변경사항 비교 후 파일 기록. (changed, diff_text) 반환."""
    if before == after:
        return False, ""
    diff_text = "\n".join(difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"before {os.path.relpath(path, PROJECT_ROOT)}",
        tofile=f"after {os.path.relpath(path, PROJECT_ROOT)}",
        lineterm=""
    ))
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(after)
    return True, diff_text

def create_commit_msg(diff: str) -> str:
    """LLM으로 커밋 메시지 생성"""
    commit_msg_prompt = PromptTemplate(
            input_variables=["diff"],
            template=(
                "다음 변경사항(diff)을 요약해 한 줄짜리 영어 커밋 메시지를 작성하세요. "
                "불필요한 설명 없이 메시지만 출력:\n\n{diff}"
            )
        )
    chain_commit_msg = commit_msg_prompt | commit_subject_llm
    subject_obj: CommitSubject = chain_commit_msg.invoke({
        "diff": "\n".join(diff)
    })
    return(subject_obj.subject)

def git_add_commit_if_changed(rel_path: str, commit_message: str) -> bool:
    # git add 
    subprocess.run(["git", "add", rel_path], cwd=GIT_WORKDIR, check=True)
    # 변경 파일 유무 확인
    # diff_proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=GIT_WORKDIR, capture_output=True, text=True)
    # if diff_proc.returncode != 0:
    #     print("git diff --cached 실패:", diff_proc.stderr)
    #     return False
    # if not diff_proc.stdout.strip():
    #     print("변경사항 없음: 커밋 생략")
    #     return False

    # git commit
    c = subprocess.run(["git", "commit", "-m", commit_message], cwd=GIT_WORKDIR)
    if c.returncode != 0:
        print("git commit 실패:", c.returncode)
        return False

def git_push():
    # push
    p = subprocess.run(["git", "push", "origin", TARGET_BRANCH], cwd=GIT_WORKDIR)
    if p.returncode != 0:
        print("git push 실패:", p.returncode)
        return False

    print("git push 완료")
    return True

def normalize_component_path(component: str) -> str:
    if ':' in component:
        return os.path.normpath(component.split(':', 1)[1])
    return component

def summarize_open_issues_text(raw_open_issues: dict) -> list[dict]:
    """open issues에서 필요한 column만 필터링"""
    issues = raw_open_issues.get("issues")
    return [{
            "path":         normalize_component_path(i.get("component")),
            "severity":     i.get("severity"),
            "status":       i.get("status"),
            "message":      i.get("message"),
            "startLine":    i.get("textRange").get("startLine"),
            "endLine":      i.get("textRange").get("endLine"),
        } for i in issues]

def summarize_hotspots(raw_hotspots: dict) -> list[dict]:
    """Hotspots에서 Path, Vulnerability Probability, Security Category, Status, Message, Line 필터링"""
    hotspots = raw_hotspots.get("hotspots")
    return [{
            "path":                     normalize_component_path(h.get("component")),
            "vulnerabilityProbability": h.get("vulnerabilityProbability"),
            "securityCategory":         h.get("securityCategory"),
            "status":                   h.get("status"),
            "message":                  h.get("message"),
            "line":                     h.get("line"),
        } for h in hotspots]

def json_to_dict(issue:list[str]) -> dict:
    """SonarQube 조회시 json으로 응답받으므로 변환"""
    s = "\n".join(issue)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON block not found in response content")
    json_text = s[start:end+1]
    return json.loads(json_text)

async def get_sonar_open_issues_text(sonar_server_params: StdioServerParameters) -> str:
    """Open issue 조회"""
    async with stdio_client(sonar_server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            tool = next(t for t in tools if t.name == "issues")
            page = 1 # default: 1
            page_size = 500 # 남은 open issue가 page size보다 적으면 그만큼만 조회함.
            raw_open_issues_info = await tool.ainvoke({
                "projects": [SONARQUBE_PROJECT_KEY],
                "statuses": ["OPEN"],
                "page": str(page),
                "page_size": str(page_size),
                "languages": "c",
                "severities": ["BLOCKER", "CRITICAL"], # Severity가 높은 두개 이슈만 가져온다
                "impact_software_qualities": ["MAINTAINABILITY"], # Array (MAINTAINABILITY, RELIABILITY, SECURITY)
                # severity가 높은 순으로 정렬
                "s": "SEVERITY",
                "asc": False,
            })
            raw_open_issues_info = raw_open_issues_info.strip().splitlines()
            raw_open_issues = json_to_dict(raw_open_issues_info)

            # open issues 존재하면 반환
            if raw_open_issues:
                return raw_open_issues
                
            # open issues가 더이상 없다면 빈 리스트 반환
            return []

async def get_sonar_hotspots(sonar_server_params: StdioServerParameters) -> str:
    """SonarQube Hotspots 조회"""
    async with stdio_client(sonar_server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            tool = next(t for t in tools if t.name == "hotspots")
            page = 1 # default: 1
            page_size = 100 # page size 조절하기
            raw_hotspots_info = await tool.ainvoke({
                "project_key": SONARQUBE_PROJECT_KEY,
                "page": str(page),
                "page_size": str(page_size),
            })
            raw_hotspots_info = raw_hotspots_info.strip().splitlines()
            raw_hotspots = json_to_dict(raw_hotspots_info)

            # hotspots 존재하면 반환
            if raw_hotspots:
                return raw_hotspots
                
            # hotspots가 더이상 없다면 빈 리스트 반환
            return []

def strip_to_marker(text: str, markers: list[str]) -> str:
    # 존재하는 마커들의 위치만 모아 최솟값을 선택
    idxs = [text.find(m) for m in markers]
    idxs = [i for i in idxs if i != -1]
    cut = min(idxs) if idxs else 0
    return text[cut:]

def refactor_file_worker(path, issue, project_context_json):
    print(path, "리팩토링 시작")
    try:
        with open(path, "r", encoding="utf-8") as f:
            before_code = f.read()
        
        after_code = make_refactor_from_issues(before_code, issue, path, project_context_json)
        if after_code == "":
            print(path, "변경 사항 없음")
            return False
        changed, diff_text = diff_and_maybe_write(path, before_code, after_code)
        if not changed:
            print(path, "변경 사항 없음")
            return False
        
        commit_msg = create_commit_msg(diff_text)
        git_add_commit_if_changed(path, commit_msg)
        return True
    
    except Exception as e:
        print(f"LLM 요청 실패 {e}")

def run_build(project_root: str) -> tuple[bool, str]:
    """빌드 실행 후 (성공여부, 로그) 반환"""
    subprocess.run(
        ["wsl", "cmake", ".."],
        cwd=f"{project_root}/build",
        stdout=subprocess.DEVNULL,
    )

    proc = subprocess.run(
        ["wsl", "cmake", "--build", "."],
        cwd=f"{project_root}/build",
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, log

def refactor_build(path, log, project_context_json):
    print(path, "빌드 수정 시작")
    with open(path, "r", encoding="utf-8") as f:
        before_code = f.read()

    after_code = make_refactor_from_build(before_code, log, path, project_context_json)
    if after_code == "":
        print("리팩토링 그대로")
        return False
    
    changed, diff_text = diff_and_maybe_write(path, before_code, after_code)
    if not changed:
        print("변경사항 없음")
        return False

    commit_msg = create_commit_msg(diff_text)
    git_add_commit_if_changed(path, commit_msg)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        timestamp TEXT, activity TEXT,
        file_path TEXT, status TEXT, details TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS issue_stats (
        timestamp TEXT, total_issues INTEGER, severity_distribution TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS build_log (
        timestamp TEXT, status TEXT,
        error_count INTEGER, error_files TEXT
    )""")
    conn.commit(); conn.close()

def log_activity(activity, file_path="", status="", details=""):
    conn = sqlite3.connect('agent_activity.db')
    c = conn.cursor()
    c.execute("INSERT INTO activity_log VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), activity, file_path, status, details))
    conn.commit()
    conn.close()

def log_issue_stats(remaining, severity_dist):
    conn = sqlite3.connect('agent_activity.db')
    c = conn.cursor()
    c.execute("INSERT INTO issue_stats VALUES (?, ?, ?)",
                (datetime.now().isoformat(), remaining, json.dumps(severity_dist)))
    conn.commit()
    conn.close()

def log_build_result(status, error_count, error_files):
    conn = sqlite3.connect('agent_activity.db')
    c = conn.cursor()
    c.execute("INSERT INTO build_log VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), status, error_count, json.dumps(error_files)))
    conn.commit()
    conn.close()

async def main():
    _init()
    init_db()

    # SonarQube MCP server 연결
    sonar_server_params = StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "-e", f"SONARQUBE_URL=https://sonarcloud.io",
            "-e", f"SONARQUBE_TOKEN={SONARQUBE_TOKEN}",
            "-e", f"SONARQUBE_ORGANIZATION={SONARQUBE_ORG}",
            "sapientpants/sonarqube-mcp-server:latest",
        ]
    )

    # 시그니처 요약, 이슈 조회, 리팩토링, 커밋/푸쉬 반복
    for round_idx in range(MAX_ROUNDS):
        log_activity("round_started", status="processing")
        print(f"\n=== Round {round_idx} ===")
        # 1) SonarQube Hotspots 조회
        # print("Security Hotspots 검색중 ...")
        # raw_hotspots = await get_sonar_hotspots(sonar_server_params)
        # 필요한 column만 필터링
        # hotspots_filtered = summarize_hotspots(raw_hotspots)
        # print(f"Security Hotspots {len(hotspots_filtered)}개 발견")

        # 2) SonarQube Open issues 조회
        raw_open_issues = await get_sonar_open_issues_text(sonar_server_params)
        # 필요한 column만 필터링
        issues_filtered = summarize_open_issues_text(raw_open_issues)
        
        SEVERITIES = ["BLOCKER", "CRITICAL"]
        severity_dict = {k: 0 for k in SEVERITIES}
        for issue in issues_filtered:
            sev = issue.get('severity', 'UNKNOWN')
            severity_dict[sev] = severity_dict.get(sev, 0) + 1
        log_issue_stats(len(issues_filtered), severity_dict)

        print(f"OPEN ISSUES {len(issues_filtered)}개 발견")
        
        # 필터링한 issue에 속하는 .c, .h 파일만 추출
        target_files = set(
            os.path.abspath(os.path.join(PROJECT_ROOT, i["path"]))
            for i in issues_filtered 
            if i["path"].endswith(".c") or i["path"].endswith(".h")
        )

        if not target_files:
            print("c파일과 h파일의 OPEN ISSUES, HOT SPOTS 없음. 루프 종료.")
            break

        # 3) C 프로젝트 전체 시그니처 요약 생성
        """
        프로젝트 시그니처 전체 말고 issue와 연관있는
        파일들의 시그니처만 추출하여 context로 전달
        """
        print("project signatures 수집 중...")
        related_header_files = set()
        for file_path in target_files:
            if file_path.endswith(".c"):
                related_header_files.update(get_related_header_files(file_path, PROJECT_ROOT))
        all_context_files = target_files.union(related_header_files)
        all_signatures = extract_signatures(PROJECT_ROOT, all_context_files)
        project_context_json = json.dumps(all_signatures, ensure_ascii=False)
        print("project signatures 수집 완료")

        any_changes = False
        backup_paths = {}
        
        # 4) 각 파일 리팩토링
        
        results = []
        for path in target_files:
            log_activity("refactoring_started", os.path.basename(path), "processing")
            # 현재 파일의 issue, hotspot
            issue = [i for i in issues_filtered if i.get("path") == path.split(PROJECT_ROOT+'\\', 1)[1]]
            # hotspot = [h for h in hotspots_filtered if h.get("path") == path.split(PROJECT_ROOT+'\\', 1)[1]]
            # 현재 파일의 context sinature
            # context_files = set([os.path.relpath(path, PROJECT_ROOT)])
            # if path.endswith(".c"):
            #     headers = get_related_header_files(path, PROJECT_ROOT)
            #     context_files.update([
            #         os.path.relpath(hpath, PROJECT_ROOT) for hpath in headers
            #     ])
            # relevant_signatures = [
            #     sig for sig in all_signatures if sig["file"] in context_files
            # ]
            # project_context_json = json.dumps(relevant_signatures, ensure_ascii=False)
            
            rst = refactor_file_worker(path, issue, project_context_json)
            results.append(rst)
            if rst:
                log_activity("refactoring_completed", os.path.basename(path), "success")
            else:
                log_activity("refactoring_failed", os.path.basename(path), "failed")

        any_changes = any(results)
            
        
        if not any_changes:
            print("이번 라운드에서 코드 변경이 발생하지 않았습니다.")
            if path in backup_paths:
                os.remove(backup_paths[path])
                del backup_paths[path]
            continue
        print("리팩토링 적용 완료")
        did_push = False
        while did_push != True:
            did_push = git_push()

        # build test
        ok, log = run_build(PROJECT_ROOT)

        if ok:
            log_build_result("success", 0, [])
        else:
            err_paths = parse_error_paths(log, PROJECT_ROOT)
            log_build_result("failed", len(err_paths), list(err_paths))

        while ok != True:
            target_files = set(
                os.path.abspath(os.path.join(PROJECT_ROOT, p))
                for p in err_paths
                    if p.endswith(".c")
            )
            print("make error file list\n", target_files)

            print("project signatures 수집 중...")
            related_header_files = set()
            for file_path in target_files:
                if file_path.endswith(".c"):
                    related_header_files.update(get_related_header_files(file_path, PROJECT_ROOT))
            all_context_files = target_files.union(related_header_files)
            target_files = list(target_files)
            all_signatures = extract_signatures(PROJECT_ROOT, all_context_files)
            project_context_json = json.dumps(all_signatures, ensure_ascii=False)
            print("project signatures 수집 완료")

            refactor_build(target_files[0], log, project_context_json)
            ok, log = run_build(PROJECT_ROOT)
        print("make 정상")
        #build test
        
        # 7) 한번에 git 푸시 (변경이 있는 경우만)
        did_push = False
        while did_push != True:
            did_push = git_push()

        log_activity("round_completed", status="completed")
        # SonarCloud 분석 반영 대기
        print(f"SonarCloud 분석 반영 대기 중...")
        time.sleep(WAIT_SECONDS)

        severity_dict = {k: 0 for k in SEVERITIES}
        for issue in issues_filtered:
            sev = issue.get('severity', 'UNKNOWN')
            severity_dict[sev] = severity_dict.get(sev, 0) + 1
        log_issue_stats(len(issues_filtered), severity_dict)

    print("\n리팩토링 루프 종료")




if __name__ == "__main__":
    asyncio.run(main())
