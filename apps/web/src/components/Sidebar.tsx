import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
} from "react";
import { createPortal } from "react-dom";
import {
  BookOpenText,
  ChatsCircle,
  Copy,
  Database,
  DotsThree,
  Graph,
  MagnifyingGlass,
  PencilSimple,
  Plus,
  ShieldCheck,
  SidebarSimple,
  Trash,
  X,
} from "@phosphor-icons/react";

type IconComponent = ComponentType<{ size?: number; weight?: "light" | "regular" | "fill" }>;

export interface SidebarSessionItem {
  readonly id: string;
  readonly title: string;
  readonly time: string;
}

export type SidebarWorkspace = "chat" | "governance" | "adaptation";

interface SidebarProps {
  activeWorkspace: SidebarWorkspace;
  onWorkspaceChange: (workspace: SidebarWorkspace) => void;
  activeSession: string;
  sessions?: readonly SidebarSessionItem[];
  onSessionChange: (sessionId: string) => void;
  onNewSession: () => void;
  onClose?: () => void;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
  onShowSessions: () => void;
  onRenameSession: (sessionId: string) => void;
  onDuplicateSession: (sessionId: string) => void;
  onTrashSession: (sessionId: string) => void;
  onOpenDatasets: () => void;
  datasetsOpen?: boolean;
  onSupportAction: (action: "guide") => void;
}

// A production workspace starts with the user's own sessions. Built-in fixtures
// remain available to tests and internal diagnostics, but are never offered as
// a normal navigation destination.
const defaultSessions: readonly SidebarSessionItem[] = [];

const workspaceItems: Array<{
  label: string;
  icon: IconComponent;
  action: SidebarWorkspace | "datasets";
}> = [
  { label: "对话研究", icon: ChatsCircle, action: "chat" },
  { label: "治理应用", icon: ShieldCheck, action: "governance" },
  { label: "适配能力", icon: Graph, action: "adaptation" },
  { label: "数据管理", icon: Database, action: "datasets" },
];

const supportItems: Array<{
  label: string;
  icon: IconComponent;
  action: "guide";
}> = [
  { label: "使用指南", icon: BookOpenText, action: "guide" },
];

function MenuIcon({ icon: Icon }: { icon: IconComponent }) {
  return <Icon size={19} weight="light" />;
}

function normalizeSessionQuery(value: string) {
  return value.normalize("NFKC").trim().toLocaleLowerCase("zh-CN");
}

export function Sidebar({
  activeWorkspace,
  onWorkspaceChange,
  activeSession,
  sessions = defaultSessions,
  onSessionChange,
  onNewSession,
  onClose,
  collapsed = false,
  onToggleCollapsed,
  onShowSessions,
  onRenameSession,
  onDuplicateSession,
  onTrashSession,
  onOpenDatasets,
  datasetsOpen = false,
  onSupportAction,
}: SidebarProps) {
  const asideRef = useRef<HTMLElement>(null);
  const searchButtonRef = useRef<HTMLButtonElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const floatingSearchRef = useRef<HTMLDivElement>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [compactSearch, setCompactSearch] = useState(collapsed);
  const [floatingPosition, setFloatingPosition] = useState({ left: 80, top: 132 });

  const normalizedQuery = normalizeSessionQuery(searchQuery);
  const filteredSessions = useMemo(
    () => normalizedQuery
      ? sessions.filter((session) => normalizeSessionQuery(session.title).includes(normalizedQuery))
      : sessions,
    [normalizedQuery, sessions],
  );

  const measureCompactSidebar = useCallback(() => {
    const width = asideRef.current?.getBoundingClientRect().width ?? 0;
    const visuallyCompact = collapsed || (width > 0 && width < 180);
    setCompactSearch(visuallyCompact);
    return visuallyCompact;
  }, [collapsed]);

  const updateFloatingPosition = useCallback(() => {
    const rect = searchButtonRef.current?.getBoundingClientRect();
    if (!rect) return;
    const width = Math.min(308, Math.max(240, window.innerWidth - rect.right - 24));
    setFloatingPosition({
      left: Math.min(rect.right + 10, window.innerWidth - width - 12),
      top: Math.max(12, Math.min(rect.top, window.innerHeight - 380)),
    });
  }, []);

  const openSearch = useCallback(() => {
    const isCompact = measureCompactSidebar();
    setSearchOpen(true);
    if (isCompact) {
      updateFloatingPosition();
      if (collapsed) onToggleCollapsed?.();
    }
  }, [collapsed, measureCompactSidebar, onToggleCollapsed, updateFloatingPosition]);

  const closeSearch = useCallback(() => {
    setSearchOpen(false);
    setSearchQuery("");
    searchButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!searchOpen) return;
    const frame = window.requestAnimationFrame(() => searchInputRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [compactSearch, searchOpen]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
        event.preventDefault();
        openSearch();
        return;
      }
      if (event.key === "Escape" && searchOpen) {
        event.preventDefault();
        closeSearch();
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [closeSearch, openSearch, searchOpen]);

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") {
      setCompactSearch(collapsed);
      return;
    }
    const observer = new ResizeObserver(() => {
      const isCompact = measureCompactSidebar();
      if (isCompact && searchOpen) updateFloatingPosition();
    });
    if (asideRef.current) observer.observe(asideRef.current);
    return () => observer.disconnect();
  }, [collapsed, measureCompactSidebar, searchOpen, updateFloatingPosition]);

  useEffect(() => {
    if (!searchOpen) return;
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (searchButtonRef.current?.contains(target) || floatingSearchRef.current?.contains(target)) return;
      if (!compactSearch && asideRef.current?.contains(target)) return;
      closeSearch();
    };
    const handleViewportChange = () => {
      if (compactSearch) updateFloatingPosition();
    };
    document.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("resize", handleViewportChange);
    window.addEventListener("scroll", handleViewportChange, true);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("resize", handleViewportChange);
      window.removeEventListener("scroll", handleViewportChange, true);
    };
  }, [closeSearch, compactSearch, searchOpen, updateFloatingPosition]);

  const selectSearchResult = (sessionId: string) => {
    onSessionChange(sessionId);
    closeSearch();
  };

  const searchField = (
    <div className="sidebar-search" role="search">
      <MagnifyingGlass size={17} weight="light" aria-hidden="true" />
      <input
        ref={searchInputRef}
        type="search"
        value={searchQuery}
        onChange={(event) => setSearchQuery(event.target.value)}
        placeholder="搜索会话标题"
        aria-label="搜索最近会话"
        aria-controls="recent-session-results"
      />
      <button
        type="button"
        onClick={searchQuery ? () => setSearchQuery("") : closeSearch}
        aria-label={searchQuery ? "清空会话搜索" : "关闭会话搜索"}
      >
        <X size={16} weight="light" />
      </button>
    </div>
  );

  const floatingSearch = searchOpen && compactSearch && typeof document !== "undefined"
    ? createPortal(
      <div
        ref={floatingSearchRef}
        className="sidebar-search-popover"
        role="dialog"
        aria-label="搜索最近会话"
        style={{ left: floatingPosition.left, top: floatingPosition.top }}
      >
        {searchField}
        <div className="sidebar-search-results" id="recent-session-results">
          {filteredSessions.length ? filteredSessions.map((session) => (
            <button
              className={activeSession === session.id ? "is-active" : ""}
              type="button"
              key={session.id}
              onClick={() => selectSearchResult(session.id)}
            >
              <span>{session.title}</span>
              <time>{session.time}</time>
            </button>
          )) : (
            <p className="sidebar-search-empty">没有匹配的会话</p>
          )}
        </div>
      </div>,
      document.body,
    )
    : null;

  return (
    <aside
      ref={asideRef}
      className={`sidebar ${collapsed ? "is-collapsed" : ""} ${searchOpen ? "is-searching" : ""}`}
      aria-label="项目与会话导航"
    >
      <div className="sidebar__top">
        <div className="brand-row">
          <button
            className="brand-toggle"
            type="button"
            onClick={onToggleCollapsed}
            aria-label={collapsed ? "展开项目导航" : "折叠项目导航"}
            title={collapsed ? "展开项目导航" : "折叠项目导航"}
            disabled={!onToggleCollapsed}
          >
            <img className="brand-mark" src="/assets/brand-mark.png" alt="" />
          </button>
          <div className="brand-copy">
            <strong>SocialGraph-FM</strong>
              <span>社交治理智能分析系统</span>
          </div>
          {onClose ? (
            <button className="icon-button sidebar__close" type="button" onClick={onClose} aria-label="关闭导航">
              <SidebarSimple size={20} weight="light" />
            </button>
          ) : null}
        </div>

        <button
          className="new-session-button"
          type="button"
          onClick={onNewSession}
          aria-label="新建研究会话"
        >
          <Plus size={20} weight="regular" />
          <span>新建会话</span>
        </button>
      </div>

      <div className="sidebar__scroll">
        <section className="sidebar-section" aria-labelledby="recent-sessions-title">
          <div className="sidebar-section__heading sidebar-section__heading--sessions">
            <span id="recent-sessions-title">最近会话</span>
            <button
              ref={searchButtonRef}
              className="sidebar-search-toggle"
              type="button"
              onClick={searchOpen ? closeSearch : openSearch}
              aria-label={searchOpen ? "关闭会话搜索" : "搜索最近会话"}
              aria-expanded={searchOpen}
              title="搜索会话（Ctrl K）"
            >
              {searchOpen ? <X size={17} weight="light" /> : <MagnifyingGlass size={18} weight="light" />}
            </button>
          </div>
          {searchOpen && !compactSearch ? searchField : null}
          <div
            className="session-list"
            aria-hidden={searchOpen && compactSearch ? true : undefined}
            inert={searchOpen && compactSearch ? true : undefined}
          >
            {filteredSessions.map((session) => (
              <div className={`session-row ${activeSession === session.id ? "is-active" : ""}`} key={session.id}>
                <button
                  className="session-item"
                  type="button"
                  onClick={() => onSessionChange(session.id)}
                  title={session.title}
                >
                  <span className="session-dot" aria-hidden="true" />
                  <span className="session-title">{session.title}</span>
                  <time>{session.time}</time>
                </button>
                <details className="session-menu">
                  <summary aria-label={`管理会话：${session.title}`} title="管理会话">
                    <DotsThree size={16} weight="regular" />
                  </summary>
                  <div className="session-menu__popover">
                    <button type="button" onClick={() => onRenameSession(session.id)}><PencilSimple size={15} />重命名</button>
                    <button type="button" onClick={() => onDuplicateSession(session.id)}><Copy size={15} />复制会话</button>
                    <button type="button" className="is-danger" onClick={() => onTrashSession(session.id)}><Trash size={15} />移入回收站</button>
                  </div>
                </details>
              </div>
            ))}
            {searchOpen && !compactSearch && !filteredSessions.length ? (
              <p className="sidebar-search-empty">没有匹配的会话</p>
            ) : null}
          </div>
          <button className="text-link sidebar-all" type="button" onClick={onShowSessions}>
            查看全部会话 <span aria-hidden="true">→</span>
          </button>
        </section>

        <section className="sidebar-section" aria-labelledby="workspace-title">
          <div className="sidebar-section__heading sidebar-section__heading--plain" id="workspace-title">
            工作台
          </div>
          <nav className="sidebar-menu">
            {workspaceItems.map(({ label, icon, action }) => (
              action === "datasets" ? (
                <button className={`sidebar-menu__item ${datasetsOpen ? "is-active" : ""}`} type="button" key={label} onClick={onOpenDatasets} aria-current={datasetsOpen ? "page" : undefined} aria-label={label}>
                  <MenuIcon icon={icon} /><span>{label}</span>
                </button>
              ) : (
                <button
                  className={`sidebar-menu__item ${activeWorkspace === action ? "is-active" : ""}`}
                  type="button"
                  key={label}
                  onClick={() => onWorkspaceChange(action)}
                  aria-current={activeWorkspace === action ? "page" : undefined}
                  aria-label={label}
                >
                  <MenuIcon icon={icon} /><span>{label}</span>
                </button>
              )
            ))}
          </nav>
        </section>

        <section className="sidebar-section sidebar-section--support" aria-labelledby="support-title">
          <div className="sidebar-section__heading sidebar-section__heading--plain" id="support-title">
            资源与支持
          </div>
          <nav className="sidebar-menu sidebar-menu--support">
            {supportItems.map(({ label, icon, action }) => (
              <button
                className="sidebar-menu__item"
                type="button"
                key={label}
                title={label}
                onClick={() => onSupportAction(action)}
              >
                <MenuIcon icon={icon} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </section>
      </div>

      <div className="sidebar__profile">
        <ShieldCheck size={18} weight="light" className="profile-shield" aria-hidden="true" />
        <span className="profile-avatar">R</span>
        <span className="profile-copy">
          <strong>Researcher</strong>
          <small>本地研究空间</small>
        </span>
      </div>
      {floatingSearch}
    </aside>
  );
}
