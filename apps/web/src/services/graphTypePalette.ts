const GRAPH_TYPE_COLOURS = [
  "#7867d9",
  "#4d86c6",
  "#48a69f",
  "#d18a51",
  "#5d93b8",
  "#6d9b72",
  "#8790a8",
] as const;

const SEMANTIC_TYPE_COLOUR_INDEX: Readonly<Record<string, number>> = Object.freeze({
  organization: 0,
  organisation: 0,
  institution: 0,
  person: 1,
  people: 1,
  user: 1,
  account: 1,
  "governance-account": 1,
  人员: 1,
  个人: 1,
  用户: 1,
  账号: 1,
  账户: 1,
  组织: 0,
  机构: 0,
  project: 2,
  项目: 2,
  community: 3,
  社区: 3,
  未分类: 6,
});

function hashType(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** Stable across graph versions and independent of which other types are visible. */
export function graphTypeColour(type: string): string {
  const key = type.trim().toLocaleLowerCase("zh-CN") || "未分类";
  const semanticIndex = SEMANTIC_TYPE_COLOUR_INDEX[key];
  return GRAPH_TYPE_COLOURS[semanticIndex ?? hashType(key) % GRAPH_TYPE_COLOURS.length];
}

export { GRAPH_TYPE_COLOURS };
