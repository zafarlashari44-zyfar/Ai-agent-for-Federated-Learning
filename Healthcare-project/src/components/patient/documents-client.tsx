"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Download, FileText, Search, Trash2, Upload } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import type { Database } from "@/types/database";
import { cn, formatDate } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type DocumentRow = Database["public"]["Tables"]["documents"]["Row"];

const categories = [
  "all",
  "lab",
  "report",
  "imaging",
  "prescription",
  "administrative",
];

const categoryVariant = {
  lab: "info",
  report: "secondary",
  imaging: "purple",
  administrative: "secondary",
  prescription: "success",
} as const;

function formatBytes(bytes: number | null) {
  if (bytes === null) return "Unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function safeFileName(name: string) {
  return name.replace(/[^a-zA-Z0-9._-]/g, "-");
}

export function DocumentsClient({
  documents,
  patientId,
  userId,
}: {
  documents: DocumentRow[];
  patientId: string;
  userId: string;
}) {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [uploadCategory, setUploadCategory] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const filtered = documents.filter((document) => {
    const query = search.toLowerCase();
    return (
      (document.name.toLowerCase().includes(query) ||
        document.type.toLowerCase().includes(query)) &&
      (category === "all" || document.type === category)
    );
  });

  async function uploadDocument() {
    if (!file || !name.trim() || !uploadCategory) {
      setMessage("Choose a file, enter a name, and select a category.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setMessage("The file must be 20 MB or smaller.");
      return;
    }

    setBusy(true);
    setMessage("");
    const supabase = createClient();
    const storagePath = `${userId}/${crypto.randomUUID()}-${safeFileName(file.name)}`;
    const { error: storageError } = await supabase.storage
      .from("patient-documents")
      .upload(storagePath, file, {
        contentType: file.type || undefined,
        upsert: false,
      });

    if (storageError) {
      setBusy(false);
      setMessage(storageError.message);
      return;
    }

    const { error: databaseError } = await supabase.from("documents").insert({
      patient_id: patientId,
      uploaded_by: userId,
      name: name.trim(),
      type: uploadCategory,
      storage_path: storagePath,
      mime_type: file.type || null,
      size_bytes: file.size,
    });

    if (databaseError) {
      await supabase.storage.from("patient-documents").remove([storagePath]);
      setBusy(false);
      setMessage(databaseError.message);
      return;
    }

    setBusy(false);
    setOpen(false);
    setName("");
    setUploadCategory("");
    setFile(null);
    router.refresh();
  }

  async function downloadDocument(document: DocumentRow) {
    setMessage("");
    const supabase = createClient();
    const { data, error } = await supabase.storage
      .from("patient-documents")
      .createSignedUrl(document.storage_path, 60, { download: document.name });

    if (error) {
      setMessage(error.message);
      return;
    }

    window.location.assign(data.signedUrl);
  }

  async function deleteDocument(document: DocumentRow) {
    if (!window.confirm(`Delete "${document.name}"?`)) {
      return;
    }

    setMessage("");
    const supabase = createClient();
    const { error: storageError } = await supabase.storage
      .from("patient-documents")
      .remove([document.storage_path]);
    if (storageError) {
      setMessage(storageError.message);
      return;
    }

    const { error } = await supabase.from("documents").delete().eq("id", document.id);
    if (error) {
      setMessage(error.message);
      return;
    }

    router.refresh();
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">My Documents</h2>
          <p className="text-sm text-gray-500">{documents.length} files</p>
        </div>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="gap-2 bg-green-600 hover:bg-green-700">
              <Upload className="h-4 w-4" /> Upload Document
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader><DialogTitle>Upload Document</DialogTitle></DialogHeader>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label>Document Name</Label>
                <Input value={name} onChange={(event) => setName(event.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Category</Label>
                <Select value={uploadCategory} onValueChange={setUploadCategory}>
                  <SelectTrigger><SelectValue placeholder="Select category" /></SelectTrigger>
                  <SelectContent>
                    {categories.slice(1).map((item) => (
                      <SelectItem key={item} value={item}>
                        {item.replaceAll("_", " ")}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <input
                ref={fileRef}
                type="file"
                className="hidden"
                accept=".pdf,.jpg,.jpeg,.png,.webp,.doc,.docx"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="w-full border-2 border-dashed border-gray-200 rounded-xl p-8 text-center hover:border-green-300"
              >
                <Upload className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                <p className="text-sm text-gray-500">
                  {file ? file.name : "Click to select a file"}
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  PDF, JPG, PNG, WEBP, DOC, DOCX up to 20 MB
                </p>
              </button>
              {message && <p className="text-sm text-red-600">{message}</p>}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
              <Button onClick={uploadDocument} disabled={busy} className="bg-green-600 hover:bg-green-700">
                {busy ? "Uploading..." : "Upload"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {message && !open && (
        <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600">{message}</p>
      )}

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <Input
            placeholder="Search documents"
            className="pl-8 bg-white"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <div className="flex gap-1 flex-wrap">
          {categories.map((item) => (
            <Button
              key={item}
              variant={category === item ? "default" : "outline"}
              size="sm"
              onClick={() => setCategory(item)}
              className={cn(
                "text-xs capitalize",
                category === item && "bg-green-600 hover:bg-green-700",
              )}
            >
              {item}
            </Button>
          ))}
        </div>
      </div>

      {filtered.length ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((document) => (
            <Card key={document.id} className="hover:shadow-md transition-all group">
              <CardContent className="p-4">
                <div className="flex items-start gap-3 mb-3">
                  <div className="p-2.5 bg-red-50 rounded-lg">
                    <FileText className="h-5 w-5 text-red-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-gray-900 truncate">
                      {document.name}
                    </p>
                    <p className="text-xs text-gray-400">{formatBytes(document.size_bytes)}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <Badge
                    variant={
                      categoryVariant[document.type as keyof typeof categoryVariant] ??
                      "secondary"
                    }
                    className="capitalize"
                  >
                    {document.type}
                  </Badge>
                  <span className="text-xs text-gray-400">
                    {formatDate(document.created_at)}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mb-3">
                  Uploaded by {document.uploaded_by === userId ? "you" : "your care team"}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 gap-1.5 h-7 text-xs"
                    onClick={() => downloadDocument(document)}
                  >
                    <Download className="h-3 w-3" /> Download
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-red-400 hover:text-red-600"
                    onClick={() => deleteDocument(document)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="p-10 text-center text-sm text-gray-500">
            {documents.length ? "No documents match the filters." : "No documents uploaded yet."}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
