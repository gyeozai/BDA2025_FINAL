import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import math

# ==============================================================================
# 函式定義區
# 將所有函式定義放在腳本的開頭
# ==============================================================================

public_data = "public_data.csv"
private_data = "private_data.csv" 
public_submission = "public_submission.csv" # public_submission_117
private_submission = "private_submission.csv" # private_submission_117

def plot_scatter_matrix(input_file, output_file):
    """
    繪製散點圖矩陣，顯示所有維度之間的關係
    
    Parameters:
    input_file (str): 輸入CSV檔案路徑
    output_file (str): 輸出圖片檔案路徑
    """
    
    # 讀取資料
    df = pd.read_csv(input_file, index_col=0)
    
    # 獲取維度資訊
    dimensions = df.columns.tolist()
    n_dims = len(dimensions)
    
    # 計算需要的子圖數量 (n choose 2)
    n_plots = n_dims * (n_dims - 1) // 2
    
    # 計算子圖排列 (一排放三張)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    
    # 創建圖形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # 如果只有一行或一個圖，確保 axes 是 2D 陣列以便索引
    if n_plots == 0:
        print(f"檔案 {input_file} 中沒有足夠的維度來繪製散點圖。")
        return
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_plots == 1:
        axes = np.array([[axes]])
    
    # 生成所有維度組合
    dim_combinations = list(combinations(dimensions, 2))
    
    # 繪製每個散點圖
    plot_idx = 0
    for i, (dim1, dim2) in enumerate(dim_combinations):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        # 繪製散點圖
        ax.scatter(df[dim1], df[dim2], alpha=0.6, s=1, color='steelblue', rasterized=True)
        
        # 設定標籤和標題
        ax.set_xlabel(f'Dim {dim1}')
        ax.set_ylabel(f'Dim {dim2}')
        ax.set_title(f'Dim {dim1} vs {dim2}')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 隱藏多餘的子圖
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # 調整佈局
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"圖片已儲存至: {output_file}")


def plot_clustering_results(data_file, submission_file, output_file, n_cols=3):
    """
    繪製分群結果的散點圖矩陣，用不同顏色顯示不同群組
    
    Parameters:
    data_file (str): 原始資料CSV檔案路徑
    submission_file (str): 分群結果CSV檔案路徑 (id, label格式)
    output_file (str): 輸出圖片檔案路徑
    n_cols (int): 每行的子圖數量
    """
    
    # 讀取原始資料和分群結果
    df_data = pd.read_csv(data_file, index_col=0)
    df_labels = pd.read_csv(submission_file)
    
    # 確保資料對齊
    df_labels.set_index('id', inplace=True)
    
    # 合併資料和標籤
    df_merged = df_data.join(df_labels, how='inner')
    
    # 獲取維度資訊
    dimensions = [col for col in df_data.columns if col != 'label']
    n_dims = len(dimensions)
    
    # 計算需要的子圖數量
    n_plots = n_dims * (n_dims - 1) // 2
    n_rows = math.ceil(n_plots / n_cols)
    
    # 創建圖形
    if n_plots == 0:
        print(f"檔案 {data_file} 中沒有足夠的維度來繪製散點圖。")
        return
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # 處理軸陣列格式
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_plots == 1:
        axes = np.array([[axes]])
    
    # 生成所有維度組合
    dim_combinations = list(combinations(dimensions, 2))
    
    # 獲取獨特的標籤並為每個標籤分配顏色
    unique_labels = sorted(df_merged['label'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    # 繪製每個散點圖
    plot_idx = 0
    for dim1, dim2 in dim_combinations:
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        # 為每個群組繪製散點
        for label in unique_labels:
            mask = df_merged['label'] == label
            subset = df_merged[mask]
            
            ax.scatter(subset[dim1], subset[dim2], 
                       alpha=0.6, 
                       s=1, 
                       color=color_map[label],
                       label=f'Cluster {label}')
        
        # 設定標籤和標題
        ax.set_xlabel(f'Dim {dim1}')
        ax.set_ylabel(f'Dim {dim2}')
        ax.set_title(f'Dim {dim1} vs {dim2} - Clustering')
        ax.grid(True, alpha=0.3)
        
        # 只在第一個子圖顯示圖例
        if plot_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_idx += 1
    
    # 隱藏多餘的子圖
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # 調整佈局
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"\n分群結果圖片已儲存至: {output_file}")
    print(f"總共有 {len(unique_labels)} 個群組: {unique_labels}")
    
    # 顯示每個群組的統計資訊
    for label in unique_labels:
        count = (df_merged['label'] == label).sum()
        print(f"群組 {label}: {count} 個點")

def main_raw_data_scatter():
    """
    主函數 - 僅處理原始資料的散點圖矩陣
    """
    print("\n處理 public_data.csv...")
    try:
        plot_scatter_matrix(public_data, "public_data_scatter_matrix.png")
    except FileNotFoundError:
        print(f"⚠️  找不到檔案: {public_data}")
    
    print("\n處理 private_data.csv...")
    try:
        plot_scatter_matrix(private_data, "private_data_scatter_matrix.png")
    except FileNotFoundError:
        print(f"⚠️  找不到檔案: {private_data}")

def main_clustering_visualize():
    """
    主函數 - 僅視覺化分群結果
    """
    print("\n🎯 開始生成分群結果視覺化...")
    print("-" * 50)
    
    try:
        plot_clustering_results(public_data, public_submission,
                                "public_clustering_results.png")
    except FileNotFoundError as e:
        print(f"⚠️  讀取公開資料或其分群結果時出錯: {e}")
        
    print("-" * 50)

    try:
        plot_clustering_results(private_data, private_submission,
                                "private_clustering_results.png")
    except FileNotFoundError as e:
        print(f"⚠️  讀取私人資料或其分群結果時出錯: {e}")
    
    print("\n✅ 分群視覺化完成！")

def main_full_report():
    """
    主函數 - 生成完整的分群分析報告
    """
    # 設定檔案名稱
    print("🔍 開始生成完整分群分析報告...")
    print("=" * 80)
    
    # 1. 原始資料散點圖
    print("\n📊 步驟 1: 生成原始資料散點圖矩陣")
    print("-" * 50)
    
    try:
        # *** 修正了變數名稱 ***
        plot_scatter_matrix(public_data, "public_data_scatter_matrix.png")
    except FileNotFoundError:
        print(f"⚠️  找不到 {public_data}")
    
    try:
        # *** 修正了變數名稱 ***
        plot_scatter_matrix(private_data, "private_data_scatter_matrix.png")
    except FileNotFoundError:
        print(f"⚠️  找不到 {private_data}")
    
    # 2. 分群結果視覺化
    print("\n🎯 步驟 2: 生成分群結果視覺化")
    print("-" * 50)
    
    try:
        # *** 修正了變數名稱 ***
        plot_clustering_results(public_data, public_submission,
                                "public_clustering_results.png")
    except FileNotFoundError as e:
        print(f"⚠️  讀取公開資料或其分群結果時出錯: {e}")
    
    try:
        # *** 修正了變數名稱 ***
        plot_clustering_results(private_data, private_submission,
                                "private_clustering_results.png")
    except FileNotFoundError as e:
        print(f"⚠️  讀取私人資料或其分群結果時出錯: {e}")
    
    # 3. 生成報告摘要
    print("\n📋 步驟 3: 生成報告摘要")
    print("-" * 50)
    
    print("\n✅ 分群分析報告生成完成！")
    print("\n📁 生成的檔案:")
    print("   1️⃣  public_data_scatter_matrix.png   - 公開資料原始分布")
    print("   2️⃣  private_data_scatter_matrix.png  - 私人資料原始分布")  
    print("   3️⃣  public_clustering_results.png    - 公開資料分群結果")
    print("   4️⃣  private_clustering_results.png   - 私人資料分群結果")
    
    print("\n💡 使用建議:")
    print("   • 將這些圖片加入到您的報告中")
    print("   • 比較原始分布和分群結果的差異")
    print("   • 分析不同維度間的關係")
    print("   • 評估分群算法的效果")
    
    print("=" * 80)


# ==============================================================================
# 程式主執行區
# 只有當這個腳本被直接執行時，才會運行以下程式碼
# ==============================================================================
if __name__ == "__main__":
    # 顯示選項菜單
    print("請選擇要執行的功能:")
    print("1. 原始資料散點圖矩陣")
    print("2. 分群結果視覺化")
    print("3. 完整分群分析報告")
    
    choice = input("請輸入選項 (1/2/3): ").strip()
    
    # 根據使用者的選擇呼叫對應的函式
    if choice == "1":
        main_raw_data_scatter()
    elif choice == "2":
        main_clustering_visualize()
    elif choice == "3":
        main_full_report()
    else:
        print("\n無效的選項。執行預設功能：完整分群分析報告...")
        main_full_report()