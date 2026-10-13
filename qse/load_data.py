import numpy as np
import cvxpy as cp
import os
import re
import pandas as pd
DATA_DIR = "data"

DATASETS = [
    "qse_dataset1_c4_mixed_asymmetric_chain.xlsx",
    "qse_dataset2_c3_mixed_symmetric_ring.xlsx",
    "qse_dataset3_c4_pure_pbr_tensor_product.xlsx",
]

def parse_complex(value):
    """
    Chuyển một ô Excel thành số phức.

    Ví dụ:
        1       -> 1 + 0j
        0.5     -> 0.5 + 0j
        "1+2j"  -> 1 + 2j
        "1-2i"  -> 1 - 2j
    """

    if pd.isna(value):
        return 0.0 + 0.0j

    if isinstance(value, (int, float, complex, np.number)):
        return complex(value)

    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("i", "j")
    text = text.replace("I", "j")

    return complex(text)


# ============================================================
# PHẦN 3. TÁCH CÁC MA TRẬN TỪ MỘT SHEET EXCEL
# ============================================================

def extract_matrix_blocks(df, n):
    """
    Tách các ma trận n x n từ DataFrame.

    - Mỗi ma trận nằm trong n hàng liên tiếp.
    - n cột đầu tiên chứa các phần tử ma trận.
    - Các ma trận được ngăn cách bởi ít nhất một hàng trống.
    """

    blocks = []
    current_block = []

    for _, row in df.iterrows():

        values = row.iloc[:n].tolist()

        # Kiểm tra hàng trống
        is_empty = all(pd.isna(v) for v in values)

        if is_empty:
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue

        # Chuyển từng phần tử sang dạng complex
        parsed_row = [parse_complex(v) for v in values]
        current_block.append(parsed_row)

    # Thêm block cuối nếu sheet không kết thúc bằng hàng trống
    if current_block:
        blocks.append(current_block)

    matrices = []

    for index, block in enumerate(blocks):

        if len(block) != n:
            raise ValueError(
                f"Ma trận thứ {index + 1} có {len(block)} hàng, "
                f"nhưng cần đúng {n} hàng."
            )

        matrix = np.array(block, dtype=complex)

        if matrix.shape != (n, n):
            raise ValueError(
                f"Ma trận thứ {index + 1} có kích thước "
                f"{matrix.shape}, không phải {(n, n)}."
            )

        matrices.append(matrix)

    return matrices


# ============================================================
# PHẦN 4. ĐỌC MỘT DATASET
# ============================================================

def load_dataset(filename):
    """
    Đọc rho_i từ Input_Rho .

    Kích thước n được lấy từ tên file, ví dụ:
    c4 -> ma trận 4 x 4
    c3 -> ma trận 3 x 3
    """

    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy file: {filepath}"
        )

    match = re.search(r"_c(\d+)_", filename)

    if match is None:
        raise ValueError(
            f"Không xác định được kích thước ma trận từ tên: {filename}"
        )

    n = int(match.group(1))

    rho_df = pd.read_excel(
        filepath,
        sheet_name="Input_Rho",
        header=None
    )



    rho_list = extract_matrix_blocks(rho_df, n)


    return n, rho_list
