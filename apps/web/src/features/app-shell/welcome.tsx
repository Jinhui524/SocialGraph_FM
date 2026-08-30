import {
  ArrowRight,
  BracketsCurly,
  CloudArrowUp,
  Graph,
  ShieldCheck,
} from "@phosphor-icons/react";
import type {
  GovernanceAssistantDispatchContext,
  GovernanceAnswerMode,
  GovernanceSkillsContext,
} from "../../types/governanceSkills";

export type ResearchPromptContextScope = "graph" | "workspace";

export interface ResearchPrompt {
  readonly icon: typeof Graph;
  readonly title: string;
  readonly text: string;
  readonly answerMode: GovernanceAnswerMode;
  readonly contextScope: ResearchPromptContextScope;
}
export const researchPrompts = Object.freeze([
  {
    icon: Graph,
    title: "图谱基本情况",
    text: "请概括当前图谱的账号规模、事实关系数量、关系类型和连通情况",
    answerMode: "overview",
    contextScope: "graph",
  },
  {
    icon: ShieldCheck,
    title: "人工复核流程",
    text: "如果要人工复核这张图，应该按什么步骤进行",
    answerMode: "review_guidance",
    contextScope: "workspace",
  },
  {
    icon: BracketsCurly,
    title: "证据核对清单",
    text: "当前图谱还需要核对哪些关系和邻域证据",
    answerMode: "evidence_requirements",
    contextScope: "workspace",
  },
] satisfies readonly ResearchPrompt[]);

export function researchPromptForText(text: string): ResearchPrompt | undefined {
  const normalized = text.trim();
  return researchPrompts.find((prompt) => prompt.text === normalized);
}

export function researchPromptDispatchRequest(
  context: GovernanceSkillsContext,
  prompt: ResearchPrompt,
): {
  readonly context: GovernanceSkillsContext;
  readonly options: GovernanceAssistantDispatchContext;
} {
  const scopedContext = prompt.contextScope === "graph"
    ? Object.freeze({ graph: context.graph, model: context.model })
    : context;
  return Object.freeze({
    context: scopedContext,
    options: Object.freeze({ intent: "answer" as const, answerMode: prompt.answerMode }),
  });
}

export function welcomePromptAction(hasGraph: boolean): "send" | "prepare_upload" {
  return hasGraph ? "send" : "prepare_upload";
}

export function WelcomeAtlas({
  onPrompt,
  onUpload,
}: {
  readonly onPrompt: (prompt: ResearchPrompt) => void;
  readonly onUpload: () => void;
}) {
  return (
    <section className="welcome-atlas" role="region" aria-label="学术网络图谱开始页">
      <div className="hero-card">
        <div className="hero-card__copy">
          <span className="hero-eyebrow">RELATIONSHIP INTELLIGENCE</span>
          <h2 id="hero-title">从关系图谱到治理研判</h2>
          <p>上传关系数据或推理包，系统将梳理网络结构、识别风险候选与协同行为，并把可核对的关系证据组织成人工复核路径。</p>
          <button className="atlas-primary-entry" type="button" onClick={onUpload}>
            <CloudArrowUp size={18} weight="bold" />上传关系数据
          </button>
        </div>
        <div className="hero-card__field" aria-label="对话研究流程">
          <span>对话研究 · 从问题到复核</span>
          <strong>描述目标，系统编排分析</strong>
          <p>系统先核对数据与分析意图，再调用图算法或 GFM；输出重点节点、群组和关系，并提供下一步复核入口。</p>
          <ol className="hero-card__route" aria-label="研究流程">
            <li>提出问题</li>
            <li>组织分析</li>
            <li>进入复核</li>
          </ol>
        </div>
      </div>

      <nav className="prompt-list" aria-label="研究入口">
        {researchPrompts.map((prompt, index) => {
          const { icon: Icon, title, text } = prompt;
          return <button type="button" className="prompt-card" key={title} onClick={() => onPrompt(prompt)}>
            <span className="prompt-index">0{index + 1}</span>
            <span className="prompt-icon"><Icon size={18} weight="light" /></span>
            <span className="prompt-card__copy"><strong>{title}</strong><small>{text}</small></span>
            <ArrowRight size={17} weight="regular" />
          </button>;
        })}
      </nav>
    </section>
  );
}
