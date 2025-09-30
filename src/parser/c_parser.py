import clang.cindex


class CParser:
    def __init__(self):
        pass

    def parse(self, file_path, project_root):
        index = clang.cindex.Index.create()
        tu = index.parse(file_path)

        result = {
            "functions_defined": [],
            "structs_defined": [],
            "functions_used": set(),
        }

        for node in tu.cursor.get_children():
            if node.location.file and node.location.file.name == file_path:
                if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    ret = node.result_type.spelling
                    fn = node.spelling
                    params = ", ".join([
                        f"{a.type.spelling} {a.spelling}".strip()
                        for a in node.get_arguments()
                    ])
                    result["functions_defined"].append(f"{ret} {fn}({params});")
                elif node.kind == clang.cindex.CursorKind.STRUCT_DECL:
                    if node.spelling:
                        result["structs_defined"].append(node.spelling)

        def find_calls(node):
            if node.kind == clang.cindex.CursorKind.CALL_EXPR:
                if node.location.file and node.location.file.name == file_path:
                    result["functions_used"].add(node.spelling)
            for child in node.get_children():
                find_calls(child)

        find_calls(tu.cursor)
        result["functions_used"] = list(result["functions_used"])
        return result