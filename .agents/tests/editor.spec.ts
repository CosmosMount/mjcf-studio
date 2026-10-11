import { test, expect } from "@playwright/test";

test("complex model: select, edit, joint preview, zero, save and reimport", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: /闭环与肌腱实验台/ }).click();
  await expect(page.locator(".project-title")).toContainText(
    "闭环与肌腱实验台",
  );
  await page.getByRole("button", { name: "gimbal_x", exact: true }).click();
  await expect(page.locator(".object-title h2")).toHaveText("gimbal_x");
  await page.getByLabel("关节 gimbal_x", { exact: true }).fill("0.2");
  await expect(page.getByLabel("数值 gimbal_x", { exact: true })).toHaveValue(
    "0.2",
  );
  await page.getByRole("button", { name: "将指定读数设为零" }).click();
  await expect(page.getByLabel("设为零的当前读数 / rad 或 m")).toHaveValue(
    "0.2",
  );
  await page.getByRole("button", { name: "应用零点标定" }).click();
  await expect(page.locator(".modal")).toHaveCount(0);
  await expect(page.getByLabel("ref", { exact: true })).toHaveValue("-0.2");
  await page.getByRole("button", { name: "撤销", exact: true }).click();
  await expect(page.getByLabel("ref", { exact: true })).toHaveValue("0");
  await page.getByRole("button", { name: "重做", exact: true }).click();
  await expect(page.getByLabel("ref", { exact: true })).toHaveValue("-0.2");
  const slider = page.getByLabel("关节 gimbal_x");
  await slider.fill("0.35");
  await page.getByRole("button", { name: "开始仿真" }).click();
  await expect(
    page.getByRole("button", { name: "暂停", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "下载工程副本", exact: true }).click();
  const download = await downloadPromise;
  const path = await download.path();
  expect(path).toBeTruthy();
  await page.getByRole("button", { name: "导入", exact: true }).click();
  await page.getByLabel("选择模型文件").setInputFiles({
    name: "saved.mjcfproj",
    mimeType: "application/zip",
    buffer: await (await import("node:fs/promises")).readFile(path!),
  });
  await page.getByRole("button", { name: "导入工作副本" }).click();
  await expect(page.locator(".modal")).toHaveCount(0);
  await page.getByRole("button", { name: "gimbal_x", exact: true }).click();
  await expect(page.getByLabel("ref", { exact: true })).toHaveValue("-0.2");
  const exported = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出 MJCF", exact: true }).click();
  expect((await exported).suggestedFilename()).toBe("mjcf-model.zip");
  expect(errors).toEqual([]);
});

test("mesh workflow: real canvas selection, preview cancellation, split, inertia and reopen", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: /网格拆分练习/ }).click();
  await page
    .getByRole("button", { name: "sample_visual", exact: true })
    .click();
  await page.getByRole("button", { name: "选择三角面", exact: true }).click();
  const canvas = page.locator("canvas");
  const bounds = (await canvas.boundingBox())!;
  await canvas.click({
    position: { x: bounds.width / 2, y: bounds.height / 2 },
  });
  await expect(page.locator(".face-toolbar strong")).not.toHaveText("0 面已选");
  await page.getByRole("button", { name: "拆分预览", exact: true }).click();
  await expect(page.locator(".modal-header h2")).toHaveText("网格拆分预览");
  await page.getByRole("button", { name: "关闭对话框" }).click();
  await expect(
    page.getByRole("button", { name: "sample_visual_split", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "拆分预览", exact: true }).click();
  await page.getByLabel("建立子刚体与转动关节").uncheck();
  await page.getByRole("button", { name: "提交拆分", exact: true }).click();
  await expect(page.locator(".modal")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "sample_visual_split", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "撤销", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "sample_visual_split", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "物理", exact: true }).click();
  await page
    .getByRole("button", { name: "计算 / 输入惯性", exact: true })
    .click();
  await page.getByRole("button", { name: "应用刚体惯性", exact: true }).click();
  await expect(page.locator(".modal")).toHaveCount(0);
  const exported = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出 MJCF", exact: true }).click();
  await exported;
  await page.screenshot({ path: "/tmp/mjcf-studio-ui.png" });
  expect(errors).toEqual([]);
});

test("gizmo drag edits the body transform", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /网格拆分练习/ }).click();
  await page.getByRole("button", { name: "sample", exact: true }).click();
  await page.getByRole("button", { name: "平移", exact: true }).click();
  const initial = await page.getByLabel("pos", { exact: true }).inputValue();
  await page.mouse.move(739, 447);
  await page.mouse.down();
  await page.mouse.move(791, 461, { steps: 12 });
  await page.mouse.up();
  await expect(page.getByLabel("pos", { exact: true })).not.toHaveValue(
    initial,
  );
  await page.getByRole("button", { name: "撤销", exact: true }).click();
  await expect(page.getByLabel("pos", { exact: true })).toHaveValue(initial);
});

test("box selection, two caps, articulated split, physical calculation and native render", async ({
  page,
}) => {
  test.setTimeout(60000);
  await page.goto("/");
  await page.getByRole("button", { name: /网格拆分练习/ }).click();
  await page
    .getByRole("button", { name: "sample_visual", exact: true })
    .click();
  await page.getByTitle("聚焦选中或全部").click();
  await page.getByLabel("观察方向").selectOption("front");
  await page.getByRole("button", { name: "框选三角面", exact: true }).click();
  await page.getByLabel("穿透", { exact: true }).check();
  const box = (await page.locator("canvas").boundingBox())!;
  // In front view, the top screen half is exactly the upper half of this subdivided cuboid.
  await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.1);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width * 0.85,
    box.y + box.height * 0.5 - 1,
    { steps: 12 },
  );
  await page.mouse.up();
  await expect(page.locator(".face-toolbar strong")).toHaveText("96 面已选");
  await page.getByRole("button", { name: "拆分预览", exact: true }).click();
  await page.getByLabel("为两侧平面边界补洞").check();
  await expect(page.locator(".mesh-diagnostics .success-text")).toHaveCount(2);
  await page.getByRole("button", { name: "提交拆分", exact: true }).click();
  await expect(page.locator(".modal")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "segment_hinge", exact: true }),
  ).toBeVisible();
  for (const name of ["sample", "segment"]) {
    await page.getByRole("button", { name, exact: true }).click();
    await page.getByRole("button", { name: "物理", exact: true }).click();
    await page
      .getByRole("button", { name: "计算 / 输入惯性", exact: true })
      .click();
    await page
      .getByRole("button", { name: "应用刚体惯性", exact: true })
      .click();
    await expect(page.locator(".modal")).toHaveCount(0);
  }
  await page.getByLabel("关节 segment_hinge").fill("0.4");
  await page.getByRole("button", { name: "渲染", exact: true }).click();
  await expect(page.locator(".render-image")).toBeVisible({ timeout: 20000 });
  await expect(page.locator(".render-image")).toHaveJSProperty(
    "naturalWidth",
    640,
  );
  await page.getByRole("button", { name: "关闭对话框" }).click();
  const result = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出 MJCF", exact: true }).click();
  await result;
});

test("display units convert authored values without changing SI storage", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /网格拆分练习/ }).click();
  await page.getByRole("button", { name: "sample", exact: true }).click();
  await page.getByRole("button", { name: "显示", exact: true }).click();
  await page.getByLabel("长度单位").selectOption("mm");
  await page.getByLabel("角度单位").selectOption("deg");
  await page.getByRole("button", { name: "关闭对话框" }).click();
  await expect(page.getByLabel("pos", { exact: true })).toHaveValue(
    "[0,0,400]",
  );
  await page.getByLabel("pos", { exact: true }).fill("[10,0,400]");
  await page.getByRole("button", { name: "应用参数", exact: true }).click();
  await page.getByRole("button", { name: "显示", exact: true }).click();
  await page.getByLabel("长度单位").selectOption("m");
  await page.getByRole("button", { name: "关闭对话框" }).click();
  await expect(page.getByLabel("pos", { exact: true })).toHaveValue(
    "[0.01,0,0.4]",
  );
});

test("mouse joint drag closes loop and rejects unreachable targets", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /闭环与肌腱实验台/ }).click();
  await page.getByRole("button", { name: "left_hinge", exact: true }).click();
  await page
    .getByRole("button", { name: "拖动关节（保持闭链）", exact: true })
    .click();
  await page.mouse.move(644, 459);
  await page.mouse.down();
  await page.mouse.move(684, 459, { steps: 12 });
  await page.mouse.up();
  await expect(page.getByLabel("数值 left_hinge", { exact: true })).toHaveValue(
    "0.4",
  );
  await expect(
    page.getByLabel("数值 right_hinge", { exact: true }),
  ).toHaveValue("0.4");
  await expect(
    page.getByLabel("数值 coupler_hinge", { exact: true }),
  ).toHaveValue("-0.4");
  const scene = await (await page.request.get("/api/scene")).json();
  expect(scene.state.constraintResidual).toBeLessThan(1e-6);
  expect(scene.state.running).toBe(false);
  await page.getByLabel("数值 coupler_hinge", { exact: true }).fill("2");
  await expect(page.getByText(/目标不可达或求解未收敛/)).toBeVisible();
  await expect(
    page.getByLabel("数值 coupler_hinge", { exact: true }),
  ).toHaveValue("-0.4");
  const after = await (await page.request.get("/api/scene")).json();
  expect(after.state.qpos).toEqual(scene.state.qpos);
  expect(after.revision).toBe(scene.revision);
});

test('joint measurement, horizontal mate, reference and zero calibration', async ({page}) => {
  await page.goto('/');
  await page.getByRole('button',{name:'导入',exact:true}).click();
  await page.getByLabel('选择模型文件').setInputFiles({name:'arm.xml',mimeType:'text/xml',buffer:Buffer.from('<mujoco><worldbody><body name="arm"><joint name="pivot" axis="0 1 0"/><geom type="capsule" size=".05" fromto="0 0 0 1 0 1"/><body pos="1 0 1"><joint name="tip" axis="0 1 0"/><geom size=".1"/></body></body></worldbody></mujoco>')});
  await page.getByRole('button',{name:'导入工作副本'}).click();
  await expect(page.locator('.modal')).toHaveCount(0);
  await page.getByRole('button',{name:'测量与水平配合',exact:true}).click();
  await page.getByLabel('测量关节 A',{exact:true}).selectOption('pivot');
  await page.getByLabel('测量关节 B',{exact:true}).selectOption('tip');
  await expect(page.getByTestId('joint-measurement')).toContainText('1414.214 mm');
  await expect(page.getByTestId('joint-measurement')).toContainText('45.0000°');
  await page.getByText('允许配合调整的关节',{exact:false}).click();
  await expect(page.getByLabel('配合调整 pivot',{exact:true})).toBeChecked();
  await expect(page.getByLabel('配合调整 tip',{exact:true})).toBeChecked();
  await page.getByLabel('配合调整 tip',{exact:true}).uncheck();
  await expect(page.getByLabel('配合调整 tip',{exact:true})).not.toBeChecked();
  await page.getByRole('button',{name:'连线水平配合',exact:true}).click();
  await expect(page.getByTestId('joint-measurement')).toContainText('0.0000°');
  await page.getByRole('button',{name:'保存水平参考姿态',exact:true}).click();
  await expect(page.getByTestId('joint-measurement')).toContainText('0.0000°');
  await page.getByRole('button',{name:'pivot',exact:true}).click();
  await page.getByRole('button',{name:'标定选中关节零点…',exact:true}).click();
  await page.getByRole('button',{name:'应用零点标定',exact:true}).click();
  await expect(page.locator('.modal')).toHaveCount(0);
  await expect(page.getByLabel('数值 pivot',{exact:true})).toHaveValue('0');
  await expect(page.getByTestId('joint-measurement')).toContainText('0.0000°');
});


test('save writes calibrated joint back to associated original XML', async ({page}) => {
 const fs=await import('node:fs/promises');
 const os=await import('node:os');
 const dir=await fs.mkdtemp(os.tmpdir()+'/mjcf-save-');
 const file=dir+'/robot.xml';
 try {
  await fs.writeFile(file,'<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>');
  await page.goto('/');
  await page.getByRole('button',{name:'打开本地文件',exact:true}).click();
  await page.getByLabel('本地文件绝对路径').fill(file);
  await page.getByRole('button',{name:'打开并关联原文件',exact:true}).click();
  await expect(page.locator('.modal')).toHaveCount(0);
  await page.getByRole('button',{name:'j',exact:true}).click();
  await page.getByRole('button',{name:'将指定读数设为零',exact:true}).click();
  await page.getByLabel('设为零的当前读数 / rad 或 m').fill('0.25');
  await page.getByRole('button',{name:'应用零点标定',exact:true}).click();
  await expect(page.locator('.modal')).toHaveCount(0);
  await page.getByRole('button',{name:'保存',exact:true}).click();
  await expect(page.locator('footer')).toContainText('已保存到 '+file);
  expect(await fs.readFile(file,'utf8')).toContain('ref="-0.25"');
 } finally { await fs.rm(dir,{recursive:true,force:true}); }
});
