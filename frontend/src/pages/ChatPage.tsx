import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Square, Loader2, ImagePlus, X, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import { Events } from "@wailsio/runtime";
import { LlmService } from "@bindings/cpm_orc";

type Msg = { role: "user" | "assistant"; text: string; image?: string };

export default function ChatPage() {
  const [status, setStatus] = useState("未加载");
  const [vision, setVision] = useState(false);
  const [modelDir, setModelDir] = useState("");
  const [locals, setLocals] = useState<string[]>([]);
  const [system, setSystem] = useState("You are a helpful assistant.");
  const [temp, setTemp] = useState("0.7");
  const [topk, setTopk] = useState("40");
  const [topp, setTopp] = useState("0.9");
  const [maxN, setMaxN] = useState("512");
  const [chatml, setChatml] = useState(true);
  const [showParams, setShowParams] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const st = await LlmService.Status();
      setStatus(st.loaded ? (st.modelType || "onnx") : "未加载");
      setVision(st.vision);
      setLocals(await LlmService.LocalOnnxModels());
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [msgs]);

  useEffect(() => {
    const offToken = Events.On("llm:token", (e: any) => {
      setMsgs((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") {
          copy[copy.length - 1] = { ...last, text: last.text + (e.data.text || "") };
        }
        return copy;
      });
    });
    const offStatus = Events.On("llm:status", (e: any) => setBusy(!!e.data.generating));
    return () => { offToken(); offStatus(); };
  }, []);

  const genOptions = () => ({
    maxNewTokens: parseInt(maxN) || 512,
    temperature: parseFloat(temp) || 0.7,
    topK: parseInt(topk) || 40,
    topP: parseFloat(topp) || 0.9,
    repetitionPenalty: 1.0,
    systemPrompt: system,
    useChatTemplate: chatml,
    seed: Date.now(),
    stream: true,
  });

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    const withImage = image;
    setInput("");
    setImage(null);
    setMsgs((m) => [...m, { role: "user", text, image: withImage || undefined }]);
    setMsgs((m) => [...m, { role: "assistant", text: "" }]);
    try {
      // 图片：交给有视觉能力的模型（当前引擎尚未支持视觉编码，仅展示附件）
      const prompt = text;
      const full = await LlmService.Generate(prompt, genOptions());
      setMsgs((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") copy[copy.length - 1] = { ...last, text: full };
        return copy;
      });
    } catch (e) {
      setMsgs((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") copy[copy.length - 1] = { ...last, text: "错误: " + e };
        return copy;
      });
    }
  };

  const pickImage = (file: File) => {
    const r = new FileReader();
    r.onload = () => {
      const b64 = r.result!.toString().split(",")[1];
      setImage(`data:image/png;base64,${b64}`);
      if (!vision) toast("当前模型不支持看图（无视觉编码器）");
    };
    r.readAsDataURL(file);
  };

  return (
    <div className="flex h-full flex-col space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">LLM 对话</h1>
          <p className="mt-1 text-sm text-muted-foreground">加载 ONNX 模型进行对话；带视觉能力的模型可直接看图。</p>
        </div>
        <Badge variant={status === "未加载" ? "secondary" : "success"}>
          {status}{vision ? " · 视觉" : ""}
        </Badge>
      </div>

      <Card>
        <CardContent className="space-y-2 pt-4">
          <div className="flex items-center gap-2">
            <Input placeholder="模型目录或本地模型 ID" value={modelDir} onChange={(e) => setModelDir(e.target.value)} />
            <Select value="" onValueChange={(v) => v && setModelDir(v)}>
              <SelectTrigger className="w-44"><SelectValue placeholder="本地模型…" /></SelectTrigger>
              <SelectContent>
                {locals.map((d) => <SelectItem key={d} value={d}>{d}</SelectItem>)}
              </SelectContent>
            </Select>
            <Button onClick={async () => { try { await LlmService.Load(modelDir); await refresh(); toast("模型加载成功"); } catch (e) { toast(String(e), true); } }}>加载</Button>
            <Button variant="outline" onClick={async () => { await LlmService.Unload(); await refresh(); }}>卸载</Button>
            <Button size="icon" variant="ghost" onClick={() => setShowParams(!showParams)}><Settings2 className="h-4 w-4" /></Button>
          </div>
          {showParams && (
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1"><Label>系统提示词</Label><Input value={system} onChange={(e) => setSystem(e.target.value)} /></div>
              <div className="grid grid-cols-4 gap-2">
                {([
                  ["温度", temp, setTemp], ["Top-K", topk, setTopk], ["Top-P", topp, setTopp], ["最大长度", maxN, setMaxN],
                ] as const).map(([label, val, set]) => (
                  <div key={label} className="space-y-1"><Label>{label}</Label><Input value={val} onChange={(e) => set(e.target.value)} /></div>
                ))}
              </div>
              <div className="flex items-center gap-2 pt-6"><Switch checked={chatml} onCheckedChange={setChatml} /><Label>ChatML 模板</Label></div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-1 flex-col">
        <CardContent className="flex min-h-0 flex-1 flex-col pt-4">
          <div ref={logRef} className="mb-3 flex-1 space-y-3 overflow-y-auto pr-1">
            {msgs.length === 0 && <p className="text-sm text-muted-foreground">开始对话吧</p>}
            {msgs.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] whitespace-pre-wrap break-words rounded-xl px-3.5 py-2.5 text-sm ${m.role === "user" ? "bg-primary text-primary-foreground" : "border bg-muted"}`}>
                  {m.image && <img src={m.image} className="mb-2 block max-h-40 max-w-[220px] rounded-md" alt="attached" />}
                  {m.text || (m.role === "assistant" && <span className="opacity-60">…</span>)}
                </div>
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {image && (
              <div className="flex items-center gap-2">
                <img src={image} className="h-12 rounded-md" alt="preview" />
                <Button size="icon" variant="ghost" onClick={() => setImage(null)}><X className="h-4 w-4" /></Button>
                <span className="text-xs text-muted-foreground">{vision ? "模型可看图，将作为附件发送" : "当前模型不支持看图"}</span>
              </div>
            )}
            <div className="flex items-center gap-2">
              <label className="cursor-pointer">
                <Button asChild variant="outline" size="icon"><span><ImagePlus className="h-4 w-4" /></span></Button>
                <input type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && pickImage(e.target.files[0])} />
              </label>
              <Input placeholder="输入消息，回车发送" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !busy && send()} />
              <Button onClick={send} disabled={busy}><Send className="h-4 w-4" />发送</Button>
              <Button variant="destructive" size="icon" onClick={() => LlmService.Stop()}><Square className="h-4 w-4" /></Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}