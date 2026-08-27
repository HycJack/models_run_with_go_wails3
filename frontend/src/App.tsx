import { useState } from "react";
import {
  Boxes,
  ScanText,
  Mic,
  MessageSquareText,
  Settings,
  ScanBarcode,
  Percent,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useToasts } from "@/lib/toast";
import ModelsPage from "@/pages/ModelsPage";
import OcrPage from "@/pages/OcrPage";
import AsrPage from "@/pages/AsrPage";
import ChatPage from "@/pages/ChatPage";
import RuntimePage from "@/pages/RuntimePage";
import MathPage from "@/pages/MathPage";

const NAV = [
  { key: "models", label: "模型管理", icon: Boxes },
  { key: "ocr", label: "PaddleOCR", icon: ScanText },
  { key: "asr", label: "语音识别", icon: Mic },
  { key: "chat", label: "LLM 对话", icon: MessageSquareText },
  { key: "math", label: "公式助手", icon: Percent },
  { key: "runtime", label: "运行环境", icon: Settings },
] as const;

type TabKey = (typeof NAV)[number]["key"];

export default function App() {
  const [tab, setTab] = useState<TabKey>("models");
  const toasts = useToasts();

  return (
    <div className="flex h-full">
      {/* Sidebar: top padding keeps the brand clear of the macOS traffic
          lights (hidden inset title bar). */}
      <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-muted px-3 pb-4 pt-14">
        <div className="mb-5 flex items-center gap-2.5 px-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent text-xs font-extrabold text-white">
            CPM
          </div>
          <div>
            <div className="text-sm font-bold leading-tight">OCR Studio</div>
            <div className="text-[11px] text-muted-foreground">ONNX · Wails</div>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          {NAV.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground",
                tab === key && "bg-secondary font-medium text-foreground shadow-[inset_2px_0_0_0_#5b8cff]"
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2 border-t border-border px-2 pt-3 text-xs text-muted-foreground">
          <ScanBarcode className="h-3.5 w-3.5 text-emerald-400" />
          后台运行 · 托盘 / 快捷键
        </div>
      </aside>

      {/* Content */}
      <main className="flex-1 overflow-y-auto px-7 py-6">
        {tab === "models" && <ModelsPage />}
        {tab === "ocr" && <OcrPage />}
        {tab === "asr" && <AsrPage />}
        {tab === "chat" && <ChatPage />}
        {tab === "math" && <MathPage />}
        {tab === "runtime" && <RuntimePage />}
      </main>

      {/* Toasts */}
      <div className="pointer-events-none fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cn(
              "pointer-events-auto rounded-lg border bg-popover px-4 py-2.5 text-sm shadow-lg",
              t.error ? "border-destructive text-destructive" : "border-border"
            )}
          >
            {t.message}
          </div>
        ))}
      </div>
    </div>
  );
}