import { GovernanceWorkspaceProvider } from "./components/GovernanceWorkspaceProvider";
import { SocialGraphApp } from "./features/app-shell/SocialGraphApp";

export * from "./features/app-shell/SocialGraphApp";

export default function App() {
  return (
    <GovernanceWorkspaceProvider>
      <SocialGraphApp />
    </GovernanceWorkspaceProvider>
  );
}
