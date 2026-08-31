import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, "../../..");
const catalogPath = path.join(repositoryRoot, "skills/governance/catalog.json");
const assistantCatalogPath = path.join(repositoryRoot, "skills/assistant/catalog.json");
const outputPath = path.join(repositoryRoot, "apps/web/src/generated/governanceSkillsContract.ts");
const check = process.argv.includes("--check");

const raw = await readFile(catalogPath, "utf8");
const catalog = JSON.parse(raw);
const assistantRaw = await readFile(assistantCatalogPath, "utf8");
const assistantCatalog = JSON.parse(assistantRaw);
if (
  catalog.namespace !== "socialgraph-fm.product-skills.governance"
  || catalog.schemaVersion !== "socialgraph-fm.governance-skills/1.0"
  || !Array.isArray(catalog.items)
  || catalog.items.length !== 8
) {
  throw new Error("unsupported SocialGraph-FM Governance canonical catalog");
}
if (
  assistantCatalog.namespace !== "socialgraph-fm.product-skills.assistant"
  || assistantCatalog.schemaVersion !== "socialgraph-fm.product-skills.assistant/1.0"
  || !Array.isArray(assistantCatalog.items)
  || assistantCatalog.items.length !== 6
) {
  throw new Error("unsupported SocialGraph-FM Assistant canonical catalog");
}
const assistantNames = assistantCatalog.items.map((item) => item.name);
const assistantPolicies = assistantCatalog.items.map((item) => ({
  name: item.name,
  label: item.label,
  description: item.description,
  uiLocation: item.uiLocation,
  readOnly: item.readOnly,
  confirmationRequired: item.confirmationRequired,
  governanceSkills: item.governanceSkills,
  parameterSchema: item.parameterSchema,
}));
if (new Set(assistantNames).size !== assistantNames.length) {
  throw new Error("SocialGraph-FM Assistant canonical catalog contains duplicate names");
}
for (const item of assistantCatalog.items) {
  if (
    typeof item.name !== "string"
    || typeof item.label !== "string"
    || typeof item.description !== "string"
    || typeof item.uiLocation !== "string"
    || item.readOnly !== true
    || item.confirmationRequired !== false
    || !Array.isArray(item.governanceSkills)
    || !item.parameterSchema
    || item.parameterSchema.type !== "object"
    || item.parameterSchema.additionalProperties !== false
  ) {
    throw new Error(`invalid Assistant canonical catalog entry: ${String(item.name)}`);
  }
}
const names = catalog.items.map((item) => item.name);
if (new Set(names).size !== names.length) {
  throw new Error("SocialGraph-FM Governance canonical catalog contains duplicate names");
}
for (const item of catalog.items) {
  if (
    typeof item.name !== "string"
    || typeof item.readOnly !== "boolean"
    || typeof item.confirmationRequired !== "boolean"
    || item.readOnly === item.confirmationRequired
    || typeof item.internalCommand !== "string"
    || typeof item.parameterSchema !== "string"
  ) {
    throw new Error(`invalid canonical catalog entry: ${String(item.name)}`);
  }
}

const readOnly = catalog.items.filter((item) => item.readOnly).map((item) => item.name);
const confirmationGated = catalog.items
  .filter((item) => item.confirmationRequired)
  .map((item) => item.name);
const policies = catalog.items.map((item) => ({
  name: item.name,
  readOnly: item.readOnly,
  confirmationRequired: item.confirmationRequired,
  ...(item.confirmationAction ? { confirmationAction: item.confirmationAction } : {}),
  internalCommand: item.internalCommand,
  parameterSchema: item.parameterSchema,
}));
const versionRoot = path.dirname(catalogPath);
const parameterSchemas = new Map();
for (const item of catalog.items) {
  const schemaPath = path.resolve(versionRoot, item.parameterSchema);
  if (!schemaPath.startsWith(`${versionRoot}${path.sep}`)) {
    throw new Error(`parameter schema escaped the canonical version root: ${item.name}`);
  }
  const schema = JSON.parse(await readFile(schemaPath, "utf8"));
  if (schema.type !== "object" || schema.additionalProperties !== false) {
    throw new Error(`parameter schema is not a strict object: ${item.name}`);
  }
  parameterSchemas.set(item.name, schema);
}

function resolveLocalReference(reference, rootSchema) {
  if (!reference.startsWith("#/$defs/")) {
    throw new Error(`unsupported parameter-schema reference: ${reference}`);
  }
  const name = reference.slice("#/$defs/".length).replaceAll("~1", "/").replaceAll("~0", "~");
  const resolved = rootSchema.$defs?.[name];
  if (!resolved) throw new Error(`missing parameter-schema definition: ${reference}`);
  return resolved;
}

function schemaType(schema, rootSchema) {
  if (schema.$ref) return schemaType(resolveLocalReference(schema.$ref, rootSchema), rootSchema);
  if (Object.hasOwn(schema, "const")) return JSON.stringify(schema.const);
  if (Array.isArray(schema.enum)) return schema.enum.map((value) => JSON.stringify(value)).join(" | ");
  if (Array.isArray(schema.anyOf)) {
    return schema.anyOf.map((value) => schemaType(value, rootSchema)).join(" | ");
  }
  if (schema.type === "null") return "null";
  if (schema.type === "string") return "string";
  if (schema.type === "integer" || schema.type === "number") return "number";
  if (schema.type === "boolean") return "boolean";
  if (schema.type === "array") return `ReadonlyArray<${schemaType(schema.items ?? {}, rootSchema)}>`;
  if (schema.type === "object") {
    const entries = Object.entries(schema.properties ?? {});
    if (!entries.length) return "Readonly<Record<string, never>>";
    const required = new Set(schema.required ?? []);
    const fields = entries.map(([name, value]) => (
      `readonly ${JSON.stringify(name)}${required.has(name) ? "" : "?"}: ${schemaType(value, rootSchema)}`
    ));
    return `{ ${fields.join("; ")} }`;
  }
  return "unknown";
}

const parameterTypeLines = catalog.items.map((item) => {
  const schema = parameterSchemas.get(item.name);
  return `  readonly ${JSON.stringify(item.name)}: ${schemaType(schema, schema)};`;
});
const sourceHash = createHash("sha256").update(raw).digest("hex");
const assistantSourceHash = createHash("sha256").update(assistantRaw).digest("hex");
const literal = (value) => JSON.stringify(value, null, 2);
const generated = `// Generated by scripts/generate-governance-skill-contract.mjs. Do not edit.\n`
  + `// Canonical catalog SHA-256: ${sourceHash}\n\n`
  + `// Assistant catalog SHA-256: ${assistantSourceHash}\n\n`
  + `export const ASSISTANT_PRODUCT_SKILL_NAMESPACE = ${JSON.stringify(assistantCatalog.namespace)} as const;\n`
  + `export const ASSISTANT_SKILLS_SCHEMA = ${JSON.stringify(assistantCatalog.schemaVersion)} as const;\n`
  + `export const ASSISTANT_SKILL_REQUEST_SCHEMA = "socialgraph-fm.assistant-skill-request/1.0" as const;\n`
  + `export const ASSISTANT_SKILL_RESULT_SCHEMA = "socialgraph-fm.assistant-skill-result/1.0" as const;\n\n`
  + `export const ASSISTANT_PUBLIC_SKILLS = ${literal(assistantNames)} as const;\n\n`
  + `export const ASSISTANT_SKILL_POLICIES = ${literal(assistantPolicies)} as const;\n\n`
  + `export const GOVERNANCE_PRODUCT_SKILL_NAMESPACE = ${JSON.stringify(catalog.namespace)} as const;\n`
  + `export const GOVERNANCE_SKILLS_SCHEMA = ${JSON.stringify(catalog.schemaVersion)} as const;\n\n`
  + `export const GOVERNANCE_PUBLIC_SKILLS = ${literal(names)} as const;\n\n`
  + `export const GOVERNANCE_READ_ONLY_SKILLS = ${literal(readOnly)} as const;\n\n`
  + `export const GOVERNANCE_CONFIRMATION_GATED_SKILLS = ${literal(confirmationGated)} as const;\n\n`
  + `export const GOVERNANCE_SKILL_POLICIES = ${literal(policies)} as const;\n\n`
  + `export type GeneratedGovernanceSkillName = typeof GOVERNANCE_PUBLIC_SKILLS[number];\n`
  + `export type GeneratedGovernanceReadOnlySkillName = typeof GOVERNANCE_READ_ONLY_SKILLS[number];\n\n`
  + `export interface GeneratedGovernanceSkillParameters {\n${parameterTypeLines.join("\n")}\n}\n`
  + `export type GeneratedGovernanceSkillParams<Name extends GeneratedGovernanceSkillName> = GeneratedGovernanceSkillParameters[Name];\n`;

if (check) {
  const existing = await readFile(outputPath, "utf8").catch(() => "");
  if (existing !== generated) {
    throw new Error("generated SocialGraph-FM Governance Web contract is stale; run npm run generate:governance-skills");
  }
} else {
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, generated, "utf8");
}
