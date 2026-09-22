"""VSCode extension: GrantScout inline completion for proposal writing.

Plain-JS extension (no build step). Provides Copilot-style inline suggestions
grounded in the user's document library via the GrantScout completion API.

Install for development:
    cd extensions/vscode-grantscout
    code --extensionDevelopmentPath=$(pwd)
Package:
    npx @vscode/vsce package
"""

const vscode = require("vscode");

/** @type {import("vscode").Disposable[]} */
let disposables = [];

function activate(context) {
  const config = () => vscode.workspace.getConfiguration("grantscout");

  const provider = {
    async provideInlineCompletionItems(document, position, _context, token) {
      const enabled = config().get("enable", true);
      if (!enabled) return undefined;

      const endpoint = (config().get("endpoint") || "").replace(/\/$/, "");
      if (!endpoint) return undefined;

      // 触发策略与 Web 版一致:词边界 + 由 VSCode 内建防抖;显式标点立即触发由
      // onDidChangeTextDocument 的手动触发命令补足(见 grantscout.trigger 命令)。
      const prefix = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
      const suffix = document.getText(new vscode.Range(position, document.positionAt(document.getText().length)));
      if (prefix.trim().length < 2) return undefined;

      const body = {
        prefix: prefix.slice(-4000),
        suffix: suffix.slice(0, 2000),
        session_id: getSessionId(),
        request_id: "r" + ++requestCounter,
        max_candidates: 1,
        corpus: config().get("corpus") || undefined,
        locale: config().get("locale") || "zh",
      };

      try {
        const response = await fetch(`${endpoint}/api/completion`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: token.isCancellationRequested ? undefined : undefined,
        });
        if (!response.ok) return undefined;
        const payload = await response.json();
        const candidate = (payload.candidates || [])[0];
        if (!candidate || !candidate.text) return undefined;
        const item = new vscode.InlineCompletionItem(candidate.text);
        item.command = {
          command: "grantscout.logAcceptance",
          title: "accept",
          arguments: [candidate.kind],
        };
        return [item];
      } catch (error) {
        return undefined;
      }
    },
  };

  disposables.push(
    vscode.languages.registerInlineCompletionItemProvider(
      [{ language: "markdown" }, { language: "plaintext" }, { scheme: "file" }],
      provider
    ),
    vscode.commands.registerCommand("grantscout.logAcceptance", (kind) => {
      // 采纳率埋点:后续可接入 /api/metrics 或 langfuse。
      lastAccepted = kind;
    }),
    vscode.commands.registerCommand("grantscout.trigger", async () => {
      await vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
    })
  );

  context.subscriptions.push(...disposables);
}

let requestCounter = 0;
let lastAccepted = null;
function getSessionId() {
  return vscode.env.machineId.slice(0, 12);
}

function deactivate() {
  disposables = [];
}

module.exports = { activate, deactivate };
