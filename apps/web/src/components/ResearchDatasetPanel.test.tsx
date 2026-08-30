import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ResearchDatasetClient } from "../services/researchDatasetClient";
import { ResearchDatasetPanel } from "./ResearchDatasetPanel";

beforeEach(() => window.localStorage.clear());
afterEach(cleanup);

const compatibleCapabilities = {
  persistentArtifacts: true,
  trustedLocalEnabled: false,
  loopbackOnly: true,
  safeUploadFormats: ["socialgraph_dataset_package"],
  runtime: {
    buildId: "test-build",
    apiContract: "socialgraph-fm-api/1.1",
    storageSchema: "dataset-store/2",
    datasetArtifactSchemas: ["2.2"],
    trainingRefSchemas: ["1.1"],
    graphHandoffSchemas: ["socialgraph-fm-graph/1.1"],
    graphFactHash: "graph-fact-hash/1",
    converterEnvironmentFingerprint: "f".repeat(64),
  },
} as const;

function lifecycleArtifact(status: "active" | "trashed" = "active") {
  return {
    schemaVersion: "2.2",
    id: "artifact-lifecycle-12345678",
    datasetName: "LifecycleGraph",
    checksum: "source",
    canonicalGraphHash: "a".repeat(64),
    contentHash: "b".repeat(64),
    manifestHash: "c".repeat(64),
    datasetRole: "target_domain",
    readinessStatus: "unchecked",
    lifecycle: status,
    scope: "complete",
    profile: { nodeCount: 12, edgeCount: 18, splitNames: [], directed: false },
    createdAt: "2026-08-11T00:00:00Z",
  } as const;
}

describe("ResearchDatasetPanel portable package selection", () => {
  it("keeps administrator-only identities out of the document until administrator tools are opened", async () => {
    const readyArtifact = {
      ...lifecycleArtifact(),
      readinessStatus: "ready",
    } as const;
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([readyArtifact]),
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);

    expect(await screen.findByText("LifecycleGraph")).toBeInTheDocument();
    expect(screen.getByText("可用")).toBeInTheDocument();
    expect(screen.queryAllByText(/TrainingDatasetRef|TrainingRef|目标域 Binding|训练合同/)).toHaveLength(0);
    expect(screen.queryAllByText(/内容哈希|图事实哈希|清单哈希|源校验和/)).toHaveLength(0);
    expect(screen.queryByText(readyArtifact.contentHash)).not.toBeInTheDocument();
    expect(screen.queryByText(readyArtifact.canonicalGraphHash)).not.toBeInTheDocument();
    expect(screen.queryByText("审计信息")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("数据目录")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("管理员工具"));
    const auditSummary = await screen.findByText("审计信息");
    const auditDetails = auditSummary.closest("details");
    expect(auditDetails).not.toHaveAttribute("open");
    expect(screen.getByLabelText("数据目录")).toBeInTheDocument();
    fireEvent.click(auditSummary);
    expect(screen.getByText("审计身份")).toBeInTheDocument();
    expect(screen.getByText(/内容哈希（contentHash）/)).toBeInTheDocument();
    expect(screen.getByText(readyArtifact.contentHash)).toBeInTheDocument();
  });

  it("keeps only data and recycle-bin views as tabs, with inference packages as an ordinary action", async () => {
    const onOpenInferencePackages = vi.fn();
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([lifecycleArtifact()]),
    } as unknown as ResearchDatasetClient;

    render(
      <ResearchDatasetPanel
        client={client}
        onOpenArtifact={vi.fn()}
        onOpenInferencePackages={onOpenInferencePackages}
      />,
    );

    expect(await screen.findAllByRole("tab")).toHaveLength(2);
    expect(await screen.findByTitle("数据服务正常")).toBeInTheDocument();
    expect(screen.queryByTitle("研究数据协议兼容")).not.toBeInTheDocument();
    expect(screen.queryByText("技术详情")).not.toBeInTheDocument();
    const activeTab = screen.getByRole("tab", { name: /我的数据 1/ });
    const recycleBinTab = screen.getByRole("tab", { name: /回收站 0/ });
    expect(activeTab.getAttribute("aria-controls")).not.toBe(recycleBinTab.getAttribute("aria-controls"));
    for (const tab of [activeTab, recycleBinTab]) {
      const panelId = tab.getAttribute("aria-controls");
      expect(panelId).toBeTruthy();
      const panel = document.getElementById(panelId!);
      expect(panel).toHaveAttribute("role", "tabpanel");
      expect(panel).toHaveAttribute("aria-labelledby", tab.id);
      if (tab.getAttribute("aria-selected") === "true") {
        expect(panel).not.toHaveAttribute("hidden");
      } else {
        expect(panel).toHaveAttribute("hidden");
      }
    }
    fireEvent.keyDown(activeTab, { key: "ArrowRight" });
    expect(recycleBinTab).toHaveFocus();
    expect(recycleBinTab).toHaveAttribute("aria-selected", "true");
    expect(document.getElementById(activeTab.getAttribute("aria-controls")!)).toHaveAttribute("hidden");
    expect(document.getElementById(recycleBinTab.getAttribute("aria-controls")!)).not.toHaveAttribute("hidden");
    fireEvent.keyDown(recycleBinTab, { key: "ArrowLeft" });
    expect(activeTab).toHaveFocus();
    expect(activeTab).toHaveAttribute("aria-selected", "true");
    fireEvent.click(recycleBinTab);
    expect(recycleBinTab).toHaveAttribute("aria-selected", "true");
    expect(document.getElementById(activeTab.getAttribute("aria-controls")!)).toHaveAttribute("hidden");
    expect(document.getElementById(recycleBinTab.getAttribute("aria-controls")!)).not.toHaveAttribute("hidden");
    fireEvent.click(activeTab);
    expect(activeTab).toHaveAttribute("aria-selected", "true");
    expect(document.getElementById(activeTab.getAttribute("aria-controls")!)).not.toHaveAttribute("hidden");
    expect(document.getElementById(recycleBinTab.getAttribute("aria-controls")!)).toHaveAttribute("hidden");
    expect(screen.queryByText(/GraphVersion|SourceArtifact|TrainingDatasetRef|GraphDatasetBinding/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "管理推理包" }));
    expect(onOpenInferencePackages).toHaveBeenCalledTimes(1);
  });

  it("does not expose a workstation path as the trusted-directory default", async () => {
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([]),
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);

    fireEvent.click(screen.getByText("管理员工具"));
    expect(await screen.findByLabelText("数据目录")).toHaveValue("");
  });

  it("shows structured candidates and re-inspects the retained file", async () => {
    const inspectPackage = vi.fn()
      .mockResolvedValueOnce({
        id: "inspection-choice",
        detectedFormat: "socialgraph_dataset_package",
        status: "mapping_required",
        issues: [{
          severity: "error",
          code: "DATASET_SELECTION_REQUIRED",
          message: "请选择数据集",
        }],
        datasetCandidates: ["alpha", "beta"],
      })
      .mockResolvedValueOnce({
        id: "inspection-beta",
        detectedFormat: "socialgraph_dataset_package",
        status: "accepted",
        issues: [],
        datasetCandidates: ["alpha", "beta"],
      });
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([]),
      inspectPackage,
      commitInspection: vi.fn(),
    } as unknown as ResearchDatasetClient;
    const { container } = render(
      <ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />,
    );
    fireEvent.click(screen.getByText("管理员工具"));
    await screen.findByLabelText("数据目录");
    const file = new File(["package"], "multi.sgfm.zip", { type: "application/zip" });
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();

    fireEvent.change(input!, { target: { files: [file] } });
    const selector = await screen.findByRole("combobox", { name: "选择包内数据集" });
    expect(selector).toHaveValue("alpha");
    fireEvent.change(selector, { target: { value: "beta" } });
    fireEvent.click(screen.getByRole("button", { name: "检查所选数据集" }));

    await waitFor(() => expect(inspectPackage).toHaveBeenCalledTimes(2));
    expect(inspectPackage).toHaveBeenNthCalledWith(1, file, undefined);
    expect(inspectPackage).toHaveBeenNthCalledWith(2, file, "beta");
    expect(await screen.findByRole("button", { name: "提交 Artifact" })).toBeInTheDocument();
  });

  it("separates a checked target-domain binding from training readiness and exposes governance blockers", async () => {
    const artifactRef = {
      schemaVersion: "2.2",
      id: "artifact-22",
      datasetName: "PrivateGraph",
      checksum: "source",
      canonicalGraphHash: "a".repeat(64),
      contentHash: "b".repeat(64),
      manifestHash: "c".repeat(64),
      datasetRole: "target_domain",
      readinessStatus: "unchecked",
      scope: "complete",
      profile: { nodeCount: 4, edgeCount: 3, splitNames: ["train_mask"], directed: false },
      createdAt: "2026-08-11T00:00:00Z",
    } as const;
    const artifact = {
      ...artifactRef,
      inspectionId: "inspection",
      sourceFormat: "graph_npz",
      sourceFiles: ["graph.npz"],
      rawManifest: {},
      derivedManifest: {},
      featureSchemas: [{ id: "feature-x", arrayName: "x", shape: [4, 4] }],
      labelSchemas: [],
      splitSets: [],
      taskSpecs: [],
      licensePolicy: { status: "unknown", identifier: "unknown", allowedUses: [], evidenceIds: ["evidence-1"] },
      licenseEvidence: [{
        id: "evidence-1",
        kind: "user_attestation",
        recordedAt: "2026-08-11T00:00:00Z",
        recordedBy: "researcher",
      }],
      dataGovernance: {
        containsPersonalData: true,
        deidentified: false,
        attributeAllowlist: [],
        excludedAttributes: ["phone"],
        retention: "project",
        userDataTrainingOptIn: false,
      },
      preparationSpec: {
        schemaVersion: "1.0",
        graphVersionId: "graph-1",
        featureAttributes: [],
        labelAttribute: null,
        taskKind: "none",
        splitStrategy: "none",
        excludedAttributes: ["phone"],
        deidentify: false,
        governance: {
          containsPersonalData: true,
          deidentified: false,
          attributeAllowlist: [],
          excludedAttributes: ["phone"],
          retention: "project",
          userDataTrainingOptIn: false,
        },
      },
      graphView: {
        id: "view",
        nodes: [],
        edges: [],
        summary: {
          nodeCount: 4,
          edgeCount: 3,
          density: 0.5,
          connectedComponents: 1,
          visibleNodeCount: 0,
          visibleEdgeCount: 0,
          partialPreview: false,
        },
      },
    } as const;
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([artifactRef]),
      getArtifact: vi.fn().mockResolvedValue(artifact),
      getReadiness: vi.fn().mockResolvedValue({
        artifactId: artifact.id,
        status: "blocked",
        contentHash: artifact.contentHash,
        manifestHash: artifact.manifestHash,
        blockers: [
          { code: "LICENSE_UNRESOLVED", message: "许可证用途尚未确认。", severity: "blocker" },
          { code: "TASK_SPEC_MISSING", message: "缺少训练任务合同。", severity: "blocker" },
        ],
        warnings: [],
        checkedAt: "2026-08-11T00:00:00Z",
      }),
    } as unknown as ResearchDatasetClient;
    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);

    expect(await screen.findByText("PrivateGraph")).toBeInTheDocument();
    expect(screen.getByText("尚未检查")).toBeInTheDocument();
    expect(screen.queryByText("审计信息")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("管理员工具"));
    const auditSummary = await screen.findByText("审计信息");
    const auditDetails = auditSummary.closest("details");
    expect(auditDetails).not.toHaveAttribute("open");
    fireEvent.click(auditSummary);
    expect(screen.getByText("审计身份")).toBeInTheDocument();
    expect(screen.getByText(/内容哈希（contentHash）/)).toBeInTheDocument();
    expect(screen.getByText(/图事实哈希（canonicalGraphHash）/)).toBeInTheDocument();
    expect(screen.getByText(artifact.contentHash)).toBeInTheDocument();
    expect(screen.getByText(artifact.canonicalGraphHash)).toBeInTheDocument();
    fireEvent.click(screen.getByText("更多操作"));
    fireEvent.click(screen.getByRole("button", { name: "检查状态" }));

    expect(await screen.findByText(/许可证用途尚未确认/)).toBeInTheDocument();
    expect(screen.getByText(/目标域 Binding 只证明图事实已交接/)).toBeInTheDocument();
    expect(screen.getByText(/含个人数据 · 未去标识/)).toBeInTheDocument();
    expect(screen.getByText("基础权重训练未授权")).toBeInTheDocument();
    expect(screen.getByText(/用户声明/)).toBeInTheDocument();
    expect(screen.getByText("阻断项（2）")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /生成 TrainingDatasetRef/ })).not.toBeInTheDocument();
  });

  it("labels and resolves a ready TrainingDatasetRef 1.1 without implying that training has started", async () => {
    const trainingRef = {
      schemaVersion: "1.1",
      artifactId: "artifact-ready",
      contentHash: "b".repeat(64),
      manifestHash: "c".repeat(64),
      graphVariant: "raw",
      splitSetId: "temporal-2019",
      featureRecipeId: "node-features-v1",
      taskSpecId: "ogbl-collab-hits50",
      datasetRole: "benchmark",
      intendedUse: "evaluation",
      refHash: "d".repeat(64),
    } as const;
    const artifactRef = {
      schemaVersion: "2.2",
      id: "artifact-ready",
      datasetName: "ogbl-collab",
      checksum: "source",
      canonicalGraphHash: "a".repeat(64),
      contentHash: "b".repeat(64),
      manifestHash: "c".repeat(64),
      datasetRole: "benchmark",
      readinessStatus: "unchecked",
      scope: "complete",
      profile: { nodeCount: 235868, edgeCount: 1285465, splitNames: ["temporal-2019"], directed: false },
      createdAt: "2026-08-11T00:00:00Z",
    } as const;
    const artifact = {
      ...artifactRef,
      inspectionId: "inspection",
      sourceFormat: "ogbl-collab",
      sourceFiles: ["graph.npz"],
      rawManifest: {},
      derivedManifest: {},
      graphVariants: [{ id: "raw" }],
      featureSchemas: [{ id: "node-features", arrayName: "x", shape: [235868, 128] }],
      labelSchemas: [],
      featureRecipes: [{ id: "node-features-v1", graphVariant: "raw", fitScope: "train" }],
      splitSets: [{ id: "temporal-2019", kind: "temporal", foldCount: 1 }],
      taskSpecs: [{ id: "ogbl-collab-hits50", kind: "link_prediction", evaluationProtocol: "Hits@50" }],
      licensePolicy: { status: "verified", identifier: "ODC-BY-1.0", allowedUses: ["evaluation"], evidenceIds: ["ogb-license"] },
      licenseEvidence: [{ id: "ogb-license", kind: "official_license", recordedAt: "2026-08-11T00:00:00Z", recordedBy: "converter" }],
      trainingRef,
      graphView: {
        id: "view",
        nodes: [],
        edges: [],
        summary: {
          nodeCount: 235868,
          edgeCount: 1285465,
          density: 0,
          connectedComponents: 1,
          visibleNodeCount: 0,
          visibleEdgeCount: 0,
          partialPreview: true,
        },
      },
    } as const;
    const resolvedHash = "e".repeat(64);
    const getMaterializedContract = vi.fn().mockResolvedValue({
      artifactId: artifact.id,
      trainingRefHash: resolvedHash,
      nodeCount: 235868,
      edgeCount: 1285465,
      featureShape: [235868, 128],
      labelShape: null,
      splitSizes: { train: 1000000, validation: 100000, test: 185465 },
      taskKind: "link_prediction",
    });
    const resolveTrainingRef = vi.fn().mockResolvedValue({
      reference: { ...trainingRef, refHash: resolvedHash },
      readiness: {
        artifactId: artifact.id,
        status: "ready",
        contentHash: artifact.contentHash,
        manifestHash: artifact.manifestHash,
        trainingRef,
        blockers: [],
        warnings: [],
        checkedAt: "2026-08-11T00:00:00Z",
      },
    });
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([artifactRef]),
      getArtifact: vi.fn().mockResolvedValue(artifact),
      getReadiness: vi.fn().mockResolvedValue({
        artifactId: artifact.id,
        status: "ready",
        contentHash: artifact.contentHash,
        manifestHash: artifact.manifestHash,
        trainingRef,
        blockers: [],
        warnings: [],
        checkedAt: "2026-08-11T00:00:00Z",
      }),
      resolveTrainingRef,
      getMaterializedContract,
    } as unknown as ResearchDatasetClient;

    const onNotify = vi.fn();
    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} onNotify={onNotify} />);
    expect(await screen.findByText("ogbl-collab")).toBeInTheDocument();
    expect(screen.queryAllByText(/TrainingDatasetRef|TrainingRef|目标域 Binding|训练合同/)).toHaveLength(0);
    expect(screen.queryByText("审计信息")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("管理员工具"));
    fireEvent.click(await screen.findByText("更多操作"));
    fireEvent.click(await screen.findByRole("button", { name: "检查状态" }));
    fireEvent.click(await screen.findByText("审计信息"));
    const resolveButton = await screen.findByRole("button", { name: "生成 TrainingDatasetRef 1.1" });
    expect(screen.getByText(/Artifact 2.2 · TrainingRef 1.1/)).toBeInTheDocument();
    fireEvent.click(resolveButton);

    await waitFor(() => expect(resolveTrainingRef).toHaveBeenCalledWith(expect.objectContaining({
      artifactId: "artifact-ready",
      splitSetId: "temporal-2019",
      splitFold: 0,
      taskSpecId: "ogbl-collab-hits50",
    })));
    await waitFor(() => expect(getMaterializedContract).toHaveBeenCalledWith("artifact-ready", resolvedHash));
    expect(screen.getByText("Fold：0")).toBeInTheDocument();
    const materialized = await screen.findByRole("region", { name: "独立物化训练合同" });
    expect(within(materialized).getByText("只读物化 Bundle 已验证")).toBeInTheDocument();
    expect(within(materialized).getByText(/235868 节点 · 1285465 关系/)).toBeInTheDocument();
    expect(within(materialized).getByText("235868 × 128")).toBeInTheDocument();
    expect(within(materialized).getByText(/train 1000000 · validation 100000 · test 185465/)).toBeInTheDocument();
    expect(within(materialized).getByText("链路预测")).toBeInTheDocument();
    expect(within(materialized).getByText(/不等同于目标域 Binding，也不代表训练已经启动/)).toBeInTheDocument();
    expect(await screen.findByText(`Ref ${resolvedHash.slice(0, 16)}`)).toBeInTheDocument();
    expect(onNotify).toHaveBeenCalledWith("数据准备凭据已生成并通过审计；尚未启动训练");
    expect(onNotify.mock.calls.flat().join(" ")).not.toMatch(/TrainingDatasetRef|TrainingRef|训练合同/u);
  });

  it("keeps legacy artifacts read-only and opens administrator tools before focusing the re-import path", async () => {
    const legacyRef = {
      schemaVersion: "2.1",
      id: "artifact-legacy",
      datasetName: "Cora legacy",
      checksum: "source",
      canonicalGraphHash: "a".repeat(64),
      contentHash: "b".repeat(64),
      manifestHash: "c".repeat(64),
      datasetRole: "benchmark",
      readinessStatus: "legacy",
      scope: "complete",
      profile: { nodeCount: 2708, edgeCount: 10556, splitNames: [], directed: false },
      createdAt: "2026-08-11T00:00:00Z",
    } as const;
    const client = {
      capabilities: vi.fn().mockResolvedValue({ ...compatibleCapabilities, trustedLocalEnabled: true }),
      listArtifacts: vi.fn().mockResolvedValue([legacyRef]),
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);

    expect(await screen.findByText("旧版需重导")).toBeInTheDocument();
    expect(screen.queryByText("旧版记录保持只读")).not.toBeInTheDocument();
    const adminTools = screen.getByText("管理员工具").closest("details");
    expect(adminTools).not.toHaveAttribute("open");
    fireEvent.click(screen.getByText("更多操作"));
    fireEvent.click(screen.getByRole("button", { name: "定位重新导入入口" }));
    await waitFor(() => expect(adminTools).toHaveAttribute("open"));
    expect(await screen.findByText("旧版记录保持只读")).toBeInTheDocument();
    expect(screen.getByText(/不会覆盖当前记录/)).toBeInTheDocument();
    expect(screen.getByLabelText("数据目录")).toHaveFocus();
  });

  it("renders an actionable protocol mismatch instead of a generic server failure", async () => {
    const client = {
      capabilities: vi.fn().mockRejectedValue(new Error("研究数据后端版本过旧或协议不兼容，训练合同不可用，请重启最新 API 后再试。")),
      listArtifacts: vi.fn().mockResolvedValue([]),
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);

    expect(await screen.findByText("服务协议不兼容")).toBeInTheDocument();
    expect(screen.getByTitle("数据服务需要更新")).toBeInTheDocument();
    expect(screen.queryByTitle("研究数据协议尚未就绪")).not.toBeInTheDocument();
    expect(screen.getByText(/当前页已禁用导入与数据准备操作/)).toBeInTheDocument();
    expect(screen.queryAllByText(/训练合同|TrainingDatasetRef|TrainingRef/u)).toHaveLength(0);
    fireEvent.click(screen.getByText("管理员工具"));
    expect(await screen.findByRole("button", { name: "选择研究包" })).toBeDisabled();
  });

  it("previews dependencies before a recoverable trash operation and restores from the recycle bin", async () => {
    let status: "active" | "trashed" = "active";
    const listArtifacts = vi.fn().mockImplementation(async () => [lifecycleArtifact(status)]);
    const getDeletionImpact = vi.fn().mockImplementation(async () => ({
      artifactId: lifecycleArtifact().id,
      lifecycle: status,
      blockers: [],
      dependents: [{
        kind: "embedded_training_ref",
        id: "training-ref-1",
        blocking: false,
        detail: {},
      }],
      preserved: ["dataset_store_audit"],
      impactHash: "d".repeat(64),
    }));
    const trashArtifact = vi.fn().mockImplementation(async () => {
      status = "trashed";
      return { lifecycle: { artifactId: lifecycleArtifact().id, status }, impact: await getDeletionImpact() };
    });
    const restoreArtifact = vi.fn().mockImplementation(async () => {
      status = "active";
      return { lifecycle: { artifactId: lifecycleArtifact().id, status }, impact: await getDeletionImpact() };
    });
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts,
      getDeletionImpact,
      trashArtifact,
      restoreArtifact,
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);
    expect(await screen.findByText("LifecycleGraph")).toBeInTheDocument();
    fireEvent.click(screen.getByText("更多操作"));
    fireEvent.click(screen.getByRole("button", { name: "删除" }));

    const impact = await screen.findByRole("region", { name: "LifecycleGraph 删除影响" });
    expect(within(impact).getByText(/数据准备引用/)).toBeInTheDocument();
    expect(within(impact).queryByText(/TrainingDatasetRef|TrainingRef/)).not.toBeInTheDocument();
    expect(within(impact).getByText(/执行时后端会重新扫描引用/)).toBeInTheDocument();
    fireEvent.click(within(impact).getByRole("button", { name: /移入回收站/ }));

    await waitFor(() => expect(trashArtifact).toHaveBeenCalledWith(lifecycleArtifact().id));
    fireEvent.click(screen.getByRole("tab", { name: /回收站 1/ }));
    expect(await screen.findByText("LifecycleGraph")).toBeInTheDocument();
    fireEvent.click(screen.getByText("更多操作"));
    fireEvent.click(screen.getByRole("button", { name: /恢复/ }));
    await waitFor(() => expect(restoreArtifact).toHaveBeenCalledWith(lifecycleArtifact().id));
    expect(await screen.findByText("回收站为空")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /我的数据 1/ }));
    expect(await screen.findByText("LifecycleGraph")).toBeInTheDocument();
    expect(listArtifacts).toHaveBeenCalledWith(true);
  });

  it("blocks permanent deletion when a GraphDatasetBinding is present", async () => {
    const artifact = lifecycleArtifact("trashed");
    const purgeArtifact = vi.fn();
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockResolvedValue([artifact]),
      getDeletionImpact: vi.fn().mockResolvedValue({
        artifactId: artifact.id,
        lifecycle: "trashed",
        blockers: [{
          kind: "graph_dataset_binding",
          id: "binding-1",
          blocking: true,
          detail: { graphVersionId: "graph-1" },
        }],
        dependents: [],
        preserved: [],
        impactHash: "e".repeat(64),
      }),
      purgeArtifact,
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);
    fireEvent.click(screen.getByText("管理员工具"));
    fireEvent.click(await screen.findByRole("tab", { name: /回收站 1/ }));
    fireEvent.click(screen.getByText("更多操作"));
    fireEvent.click(screen.getByRole("button", { name: "查看依赖" }));
    const impact = await screen.findByRole("region", { name: "LifecycleGraph 删除影响" });
    expect(within(impact).getByText(/GraphDatasetBinding/)).toBeInTheDocument();
    expect(within(impact).getByText(/存在阻断引用，永久删除不可用/)).toBeInTheDocument();
    fireEvent.change(within(impact).getByRole("textbox", { name: /资源 ID 后 8 位/ }), {
      target: { value: "12345678" },
    });
    expect(within(impact).getByRole("button", { name: "永久删除" })).toBeDisabled();
    expect(purgeArtifact).not.toHaveBeenCalled();
  });

  it("requires the final eight ID characters and sends the preview impactHash when purging", async () => {
    let artifacts = [lifecycleArtifact("trashed")];
    const impactHash = "f".repeat(64);
    const purgeArtifact = vi.fn().mockImplementation(async () => {
      artifacts = [];
      return { artifactId: lifecycleArtifact().id, purged: true, cleanupPending: false };
    });
    const client = {
      capabilities: vi.fn().mockResolvedValue(compatibleCapabilities),
      listArtifacts: vi.fn().mockImplementation(async () => artifacts),
      getDeletionImpact: vi.fn().mockResolvedValue({
        artifactId: lifecycleArtifact().id,
        lifecycle: "trashed",
        blockers: [],
        dependents: [],
        preserved: ["audit_log"],
        impactHash,
      }),
      purgeArtifact,
    } as unknown as ResearchDatasetClient;

    render(<ResearchDatasetPanel client={client} onOpenArtifact={vi.fn()} />);
    fireEvent.click(await screen.findByRole("tab", { name: /回收站 1/ }));
    fireEvent.click(screen.getByText("更多操作"));
    fireEvent.click(screen.getByRole("button", { name: "查看依赖" }));
    const impact = await screen.findByRole("region", { name: "LifecycleGraph 删除影响" });
    const purgeButton = within(impact).getByRole("button", { name: "永久删除" });
    expect(purgeButton).toBeDisabled();
    fireEvent.change(within(impact).getByRole("textbox", { name: /资源 ID 后 8 位/ }), {
      target: { value: "12345678" },
    });
    expect(purgeButton).toBeEnabled();
    fireEvent.click(purgeButton);

    await waitFor(() => expect(purgeArtifact).toHaveBeenCalledWith(
      lifecycleArtifact().id,
      impactHash,
      "12345678",
    ));
    expect(await screen.findByText("回收站为空")).toBeInTheDocument();
  });
});
