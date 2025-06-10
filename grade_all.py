import pandas as pd
import os
from grader import score

result_path = "result"

def load_and_score(file_path):
    try:
        submission = pd.read_csv(file_path).sort_values("id").reset_index(drop=True)
        labels_pred = submission["label"].tolist()
        return round(score(labels_pred), 4)
    except Exception as e:
        print(f"[Error] Failed to score {file_path}: {e}")
        return None

def main():
    config_path = f"{result_path}/config.txt"
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found: {config_path}")
        return

    config_df = pd.read_csv(config_path)
    
    count = 0
    total_runtime = 0.0
    results = []
    for _, row in config_df.iterrows():
        count += 1
        i = int(row["ID"])
        path = f"{result_path}/submission_{i}.csv"
        if os.path.exists(path):
            sc = load_and_score(path)
            if sc is not None:
                try:
                    runtime = float(row["Time_Seconds"])
                    total_runtime += runtime
                except:
                    runtime = None

                results.append({
                    "ID": i,
                    "Score": sc,
                    "Method": row["Method"],
                    "Scaler": row["Scaler"],
                    "Covariance": row["Covariance"],
                    "Adjacent_Focus": row["Adjacent_Focus"],
                    "Time": row["Time_Seconds"]
                })
        else:
            print(f"[Warning] File not found: {path}")

    results.sort(key=lambda x: x["Score"], reverse=True)

    print("\nScore Leaderboard (High → Low):")
    print(f"{'ID':<5} {'Score':<8} {'Method':<18} {'Scaler':<10} {'Covariance':<10} {'Adjacent':<8} {'Time(s)':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['ID']:<5} {r['Score']:<8.4f} {r['Method']:<18} {r['Scaler']:<10} {r['Covariance']:<10} {r['Adjacent_Focus']:<8} {r['Time']:<8.2f}")
    print("\nTotal submissions count:", count)
    print(f"Total runtime (seconds): {total_runtime:.2f} ({total_runtime / 3600:.2f} hrs)")

if __name__ == "__main__":
    main()
