import { Chat } from "@/components/Chat";
import { MolmoLocalizePanel } from "@/components/MolmoLocalizePanel";

export default function Home() {
  return (
    <div className="min-h-dvh">
      <MolmoLocalizePanel />
      <Chat />
    </div>
  );
}
