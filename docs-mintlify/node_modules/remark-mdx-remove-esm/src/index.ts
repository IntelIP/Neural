import type { Plugin } from "unified";
import type { Root, Node, Literal } from "mdast";
import type { MdxjsEsm } from "mdast-util-mdxjs-esm";
import { remove } from "unist-util-remove";

export type MdxEsmSpecifier = "import" | "export";

export type MdxRemoveEsmOptions = MdxEsmSpecifier | MdxEsmSpecifier[];

function isLiteral(node: Node): node is Literal {
  return "value" in node;
}

function isMdxjsEsm(node: Literal): node is MdxjsEsm {
  return node.type === "mdxjsEsm";
}

export function clsx(
  arr: (MdxEsmSpecifier | false | null | undefined | 0)[],
): MdxEsmSpecifier[] {
  return arr.filter((item): item is MdxEsmSpecifier => !!item);
}

const RemarkMdxRemoveEsm: Plugin<[MdxRemoveEsmOptions?], Root> = (options) => {
  if (typeof options === "undefined") return removeAllEsm();

  if (typeof options === "string") {
    if (options === "export") return removeExports();
    else if (options === "import") return removeImports();
  }

  if (Array.isArray(options)) {
    if (options.includes("import") && options.includes("export")) {
      return removeAllEsm();
    } else if (options.includes("export")) {
      return removeExports();
    } else if (options.includes("import")) {
      return removeImports();
    }
  }

  function removeAllEsm() {
    return (tree: Root): undefined => {
      remove(tree, "mdxjsEsm");
    };
  }

  function removeExports() {
    return (tree: Root): undefined => {
      remove(tree, (node) => {
        // type predicate
        if (!isLiteral(node)) return;

        // type predicate
        if (!isMdxjsEsm(node)) return;

        const type = node.data?.estree?.body[0].type;

        return (
          type === "ExportNamedDeclaration" ||
          type === "ExportDefaultDeclaration" ||
          type === "ExportAllDeclaration"
        );
      });
    };
  }

  function removeImports() {
    return (tree: Root): undefined => {
      // remove import declarations
      remove(tree, (node) => {
        // type predicate
        if (!isLiteral(node)) return;

        // type predicate
        if (!isMdxjsEsm(node)) return;

        const type = node.data?.estree?.body[0].type;

        return type === "ImportDeclaration";
      });
    };
  }

  return;
};

export default RemarkMdxRemoveEsm;
