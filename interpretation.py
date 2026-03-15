import io

import cairosvg
import dgl
import dgllife
import matplotlib
import matplotlib.cm as cm
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import DeepPurpose.CompoundPred as models


def build_attentivefp_from_pretrained(
    pretrained_model_name,
    node_feat_size=39,
    edge_feat_size=11,
    num_layers=3,
    graph_feat_size=64,
):
    """
    从 DeepPurpose 预训练模型中提取参数，并加载到 dgllife 的 AttentiveFPPredictor。
    """

    net = models.model_pretrained(pretrained_model_name)
    state_dict = net.model.state_dict()

    # 去掉 DeepPurpose 权重前缀
    converted_state_dict = {}
    for key, value in state_dict.items():
        converted_state_dict[key[11:]] = value

    # 删除最后四个不匹配的参数
    for _ in range(4):
        converted_state_dict.popitem()

    # 初始化 AttentiveFP 模型
    attentive_fp = dgllife.model.AttentiveFPPredictor(
        node_feat_size=node_feat_size,
        edge_feat_size=edge_feat_size,
        num_layers=num_layers,
        graph_feat_size=graph_feat_size,
    )

    # 替换预测层参数
    attentive_fp_state = attentive_fp.state_dict()
    converted_state_dict["predict.1.weight"] = attentive_fp_state["predict.1.weight"]
    converted_state_dict["predict.1.bias"] = attentive_fp_state["predict.1.bias"]

    attentive_fp.load_state_dict(converted_state_dict)
    attentive_fp.eval()

    return attentive_fp


def get_atom_colors(
    smiles,
    model,
    timestep,
    cmap_name="bwr",
    vmax=1.28,
):
    """
    根据模型得到的 atom attention weight 生成原子颜色。
    """

    g = mol_to_graph(smiles)
    g = dgl.batch([g])

    atom_feats = g.ndata["h"]
    bond_feats = g.edata["e"]

    _, atom_weights = model(g, atom_feats, bond_feats, get_node_weight=True)
    atom_weights = atom_weights[timestep]

    min_value = torch.min(atom_weights)
    max_value = torch.max(atom_weights)

    # 归一化权重
    if torch.isclose(max_value, min_value):
        atom_weights = torch.zeros_like(atom_weights)
    else:
        atom_weights = (atom_weights - min_value) / (max_value - min_value)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    atom_colors = {
        i: mapper.to_rgba(atom_weights[i].item())
        for i in range(g.number_of_nodes())
    }

    return atom_colors


def draw_smiles_image(
    smiles,
    atom_colors,
    image_size=(280, 280),
):
    """
    根据 SMILES 绘制高亮分子结构图，并返回 PIL Image。
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: {}".format(smiles))

    rdDepictor.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    drawer.SetFontSize(1)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightBonds=[],
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    # SVG 转 PNG
    png_data = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    return Image.open(io.BytesIO(png_data))


def merge_images(images, per_row=4):
    """
    将多个分子图片拼接为一张大图。
    """

    if not images:
        raise ValueError("Image list is empty.")

    widths, heights = zip(*(img.size for img in images))

    cell_width = max(widths)
    cell_height = max(heights)

    num_cols = min(per_row, len(images))
    num_rows = (len(images) + per_row - 1) // per_row

    canvas_width = cell_width * num_cols
    canvas_height = cell_height * num_rows

    merged_image = Image.new("RGB", (canvas_width, canvas_height), "white")

    for idx, img in enumerate(images):
        row = idx // per_row
        col = idx % per_row

        x_offset = col * cell_width
        y_offset = row * cell_height

        merged_image.paste(img, (x_offset, y_offset))

    return merged_image


def visualize_and_save_smiles(
    smiles_list,
    model,
    timestep,
    save_path,
    per_row=4,
    image_size=(280, 280),
):
    """
    根据 SMILES 列表生成分子图，并保存拼接后的结果。
    """

    images = []

    for smiles in smiles_list:
        try:
            atom_colors = get_atom_colors(
                smiles=smiles,
                model=model,
                timestep=timestep,
            )

            img = draw_smiles_image(
                smiles=smiles,
                atom_colors=atom_colors,
                image_size=image_size,
            )

            images.append(img)

        except Exception as exc:
            print("WARNING: Failed to process SMILES {}. Error: {}".format(smiles, exc))

    if not images:
        print("WARNING: No images were generated.")
        return

    merged_image = merge_images(images, per_row=per_row)
    merged_image.save(save_path)

    print("Image saved to {}".format(save_path))

    display(merged_image)