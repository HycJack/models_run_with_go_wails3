import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, Square, Upload, Loader2, Download, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import { AsrService } from "@bindings/cpm_orc/internal/app";

export default function AsrPage() {
  const [backend, setBackend] = useState("sensevoice");
  const [svDir, setSvDir] = useState("");
  const [whisperBin, setWhisperBin] = useState("");
  const [whisperModel, setWhisperModel] = useState("");
  const [mossBin, setMossBin] = useState("");
  const [mossModel, setMossModel] = useState("");
  const [mossBinReady, setMossBinReady] = useState(false);
  const [mossModelReady, setMossModelReady] = useState(false);
  const [ready, setReady] = useState(false);
  const [showWhisper, setShowWhisper] = useState(false);
  const [showMoss, setShowMoss] = useState(false);
  const [lang, setLang] = useState("auto");
  const [audioName, setAudioName] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [result, setResult] = useState("尚未转写");
  const [segments, setSegments] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recSec, setRecSec] = useState(0);
  const timerRef = useRef<any>(null);

  const refresh = useCallback(async () => {
    try {
      const st = await AsrService.Status();
      setBackend(st.backend);
      setSvDir(st.senseVoiceDir || "");
      setWhisperBin(st.binPath || "");
      setWhisperModel(st.modelPath || "");
      setMossBin(st.mossBin || "");
      setMossModel(st.mossModel || "");
      setMossBinReady(st.mossBinReady || false);
      setMossModelReady(st.mossModelReady || false);
      setReady(
        st.backend === "whisper" ? (st.binReady && st.modelReady) :
        st.backend === "moss" ? (st.mossBinReady && st.mossModelReady) :
        st.senseVoiceReady
      );
      setRecording(st.recording);
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const transcribe = async (bytes: string, name: string) => {
    setBusy(true);
    setResult("转写中…");
    setSegments([]);
    try {
      const text = await AsrService.TranscribeBase64(bytes, lang, name);
      setResult(text || "(空)");
      toast("转写完成");
    } catch (e) {
      setResult("转写失败: " + e);
      toast(String(e), true);
    } finally { setBusy(false); }
  };

  const run = () => {
    if (!audioFile) { toast("请先选择音频", true); return; }
    const r = new FileReader();
    r.onload = () => transcribe(r.result!.toString().split(",")[1], audioFile.name);
    r.readAsDataURL(audioFile);
  };

  const toggleRecord = async () => {
    if (recording) {
      clearInterval(timerRef.current);
      setResult("转写中…");
      try {
        const text = await AsrService.StopAndTranscribe(lang);
        setResult(text || "(空)");
        setRecording(false);
        toast("录音转写完成");
      } catch (e) { setResult("转写失败: " + e); toast(String(e), true); setRecording(false); }
    } else {
      try {
        await AsrService.StartRecording();
        setRecording(true);
        setRecSec(0);
        timerRef.current = setInterval(() => setRecSec((s) => s + 1), 1000);
      } catch (e) { toast(String(e), true); }
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">语音识别</h1>
          <p className="mt-1 text-sm text-muted-foreground">SenseVoice（ONNX）、whisper.cpp（Metal）、MOSS-Transcribe-Diarize（说话人分离，CPU）</p>
        </div>
        <Badge variant={recording ? "warning" : ready ? "success" : "secondary"}>{recording ? "录音中" : ready ? "就绪" : "未就绪"}</Badge>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="flex items-center gap-2"><Settings2 className="h-4 w-4" />引擎与模型</CardTitle>
          <Select value={backend} onValueChange={async (v) => { try { await AsrService.SetBackend(v); setBackend(v); await refresh(); } catch (e) { toast(String(e), true); } }}>
            <SelectTrigger className="w-72"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="sensevoice">SenseVoice（中文优，内置标点）</SelectItem>
              <SelectItem value="whisper">whisper.cpp（Metal）</SelectItem>
              <SelectItem value="moss">MOSS-Transcribe-Diarize（说话人分离+时间戳，CPU）</SelectItem>
            </SelectContent>
          </Select>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* SenseVoice */}
          <div className="flex items-center gap-2">
            <Label className="w-36 shrink-0">SenseVoice 目录</Label>
            <Input value={svDir} onChange={(e) => setSvDir(e.target.value)} />
            <Button variant="outline" onClick={async () => { try { await AsrService.SetSenseVoiceDir(svDir); await refresh(); toast("SenseVoice 加载成功"); } catch (e) { toast(String(e), true); } }}>加载</Button>
          </div>

          {/* whisper */}
          <Button size="sm" variant="ghost" onClick={() => setShowWhisper(!showWhisper)}>
            <Settings2 className="h-4 w-4" />whisper.cpp 配置（备用引擎）
          </Button>
          {showWhisper && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label className="w-36 shrink-0">whisper-cli</Label>
                <Input value={whisperBin} onChange={(e) => setWhisperBin(e.target.value)} />
                <Button size="sm" variant="outline" onClick={async () => { try { await AsrService.SetBin(whisperBin); toast("已保存"); } catch (e) { toast(String(e), true); } }}>设置</Button>
              </div>
              <div className="flex items-center gap-2">
                <Label className="w-36 shrink-0">whisper 模型</Label>
                <Input value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)} />
                <Button size="sm" variant="outline" onClick={async () => { try { await AsrService.SetModel(whisperModel); toast("已保存"); } catch (e) { toast(String(e), true); } }}>设置</Button>
                <Button size="sm" onClick={async () => { try { await AsrService.DownloadModel(); await refresh(); toast("模型下载完成"); } catch (e) { toast(String(e), true); } }}><Download className="h-4 w-4" />下载</Button>
              </div>
            </div>
          )}

          {/* MOSS */}
          <Button size="sm" variant="ghost" onClick={() => setShowMoss(!showMoss)}>
            <Settings2 className="h-4 w-4" />MOSS-Transcribe-Diarize 配置（CPU，无需 Python）
          </Button>
          {showMoss && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label className="w-36 shrink-0">moss-transcribe</Label>
                <Input value={mossBin} onChange={(e) => setMossBin(e.target.value)} placeholder="moss-transcribe 二进制路径" />
                <Button size="sm" variant="outline" onClick={async () => { try { await AsrService.SetMossBin(mossBin); toast("已保存"); } catch (e) { toast(String(e), true); } }}>设置</Button>
                <Badge variant={mossBinReady ? "success" : "secondary"}>{mossBinReady ? "已找到" : "未找到"}</Badge>
              </div>
              <div className="flex items-center gap-2">
                <Label className="w-36 shrink-0">GGUF 模型</Label>
                <Input value={mossModel} onChange={(e) => setMossModel(e.target.value)} placeholder="moss-transcribe-q5_k.gguf 路径" />
                <Button size="sm" variant="outline" onClick={async () => { try { await AsrService.SetMossModel(mossModel); toast("已保存"); } catch (e) { toast(String(e), true); } }}>设置</Button>
                <Button size="sm" onClick={async () => { try { await AsrService.DownloadMossModel(); await refresh(); toast("模型下载完成"); } catch (e) { toast(String(e), true); } }}><Download className="h-4 w-4" />下载</Button>
                <Badge variant={mossModelReady ? "success" : "secondary"}>{mossModelReady ? "已找到" : "未找到"}</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                先构建 CLI：<code>git clone --recursive https://github.com/mudler/moss-transcribe.cpp && cd moss-transcribe.cpp && cmake -B build && cmake --build build -j</code>
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><Mic className="h-4 w-4" />输入</CardTitle>
            <div className="flex items-center gap-2">
              <label className="cursor-pointer">
                <Button asChild variant="outline"><span><Upload className="h-4 w-4" />选择音频</span></Button>
                <input type="file" accept="audio/*" className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) { setAudioFile(f); setAudioName(f.name); } }} />
              </label>
              <Select value={lang} onValueChange={setLang}>
                <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">自动</SelectItem>
                  <SelectItem value="zh">中文</SelectItem>
                  <SelectItem value="en">英语</SelectItem>
                  <SelectItem value="ja">日语</SelectItem>
                  <SelectItem value="ko">韩语</SelectItem>
                </SelectContent>
              </Select>
              <Button variant={recording ? "destructive" : "default"} onClick={toggleRecord}>
                {recording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                {recording ? `停止 ${recSec}s` : "录音识别"}
              </Button>
              <Button variant="outline" onClick={run} disabled={busy || !audioFile}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}转写
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">{audioName ? `已选择: ${audioName}` : "选择音频文件或直接点「录音识别」"}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><Mic className="h-4 w-4" />转写结果</CardTitle>
            <Badge>{backend === "whisper" ? "whisper" : backend === "moss" ? "MOSS" : "SenseVoice"}</Badge>
          </CardHeader>
          <CardContent>
            <pre className="max-h-[360px] overflow-y-auto whitespace-pre-wrap break-all rounded-lg border bg-muted p-3 font-mono text-sm">{result}</pre>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
