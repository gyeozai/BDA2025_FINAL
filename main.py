from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
import pandas as pd
import numpy as np
import re
import argparse
import warnings
warnings.filterwarnings('ignore')


def perform_clustering(input_file, output_file,
                       method='specialized_gmm',
                       scaler_type='robust',
                       covariance_type='tied',
                       adjacent_focus=None,
                       flags=""
                      ):
    """Advanced clustering for physics event data with exponential-like distribution."""
    
    # Load the dataset
    data_df = pd.read_csv(input_file)

    # Identify ID column
    id_col = data_df.columns[0]
    if id_col.lower() == 'id':
        data_df.rename(columns={id_col: 'id'}, inplace=True)
    else:
        print(f"[Warning] No 'id' column found. Using first column '{id_col}' as ID")
        data_df.rename(columns={id_col: 'id'}, inplace=True)

    output_ids = data_df['id']
    features_df = data_df.drop(columns=['id'])
    n_dimensions = features_df.shape[1]
    
    print(f"Dataset shape: {features_df.shape}")
    print(f"Data distribution summary:")
    print(features_df.describe())
    
    # Choose number of clusters based on method and hint
    k_clusters = 4 * n_dimensions - 1
    
    print(f"Target clusters: {k_clusters}")
    if adjacent_focus:
        print(f"Focusing on adjacent dimensions: {adjacent_focus}")

    # 1. (Optional) Specialized Feature Engineering for Physics Data
    if "feature" in flags:
        print("[FEATURE] Feature engineering ENABLED.")
        processed_features = engineer_physics_features(features_df, method)
    else:
        processed_features = features_df
        
    # 2. Specialized Scaling for Exponential-like Distribution
    scaled_features = apply_specialized_scaling(processed_features, scaler_type)
    print(f"Features scaled using specialized {scaler_type} approach.")
    
    # 3. (Optional) Outlier Handling - mark and potentially weight differently
    # Note: You must modify other parts of the code to support this.
    if "outlier" in flags:
        print("[OUTLIER] Outlier handling ENABLED (not yet implemented).")
        # outlier_weights = detect_and_weight_outliers(scaled_features)

    # 4. Dimensionality and Focus Strategy
    if method == 'adjacent_focus':
        # Focus on specified adjacent dimensions
        final_features = focus_on_adjacent_dims(scaled_features, features_df.columns, adjacent_focus)
    elif method == 'hybrid':
        # Combine original features with adjacent dimension emphasis
        final_features = create_hybrid_features(scaled_features, features_df.columns, adjacent_focus)
    else:
        final_features = scaled_features
    
    # 5. Specialized Clustering
    print(f"Fitting {method.upper()} model...")

    if method == 'hierarchical':
        cluster_labels = perform_hierarchical_clustering(k_clusters, adjacent_focus, final_features)
        model = None
    else:
        if method == 'hybrid':
            model = create_hybrid_model(k_clusters, covariance_type)
        elif method == 'adjacent_focus':
            model = create_adjacent_focused_model(k_clusters, covariance_type)
        elif method == 'specialized_gmm':
            model = create_specialized_gmm(k_clusters, covariance_type, final_features)
        elif method == 'specialized_gmm_v0':
            model = create_specialized_gmm_v0(k_clusters, covariance_type, final_features)
        elif method == 'gmm':
            model = create_gmm_model(k_clusters, covariance_type)
        else:
            model = create_kmeans_model(k_clusters)

        # Fit model
        model.fit(final_features)
        cluster_labels = model.predict(final_features)

    # 6. (Optional) Post-processing: Refine clusters based on physics intuition
    if "refine" in flags:
        print("[REFINE] Physics-based refinement ENABLED.")
        cluster_labels = refine_clusters_physics_aware(cluster_labels, features_df)

    # 7. Create submission
    output_df = pd.DataFrame({
        'id': output_ids,
        'label': cluster_labels
    })
    output_df.to_csv(output_file, index=False)
    print(f"Submission saved to '{output_file}'")
    print(f"Cluster distribution:\n{output_df['label'].value_counts().sort_index()}")
    
    return model, final_features

def engineer_physics_features(features_df, method):
    """Create physics-aware features"""
    
    features = features_df.copy()
    
    if method in ['specialized_gmm', 'specialized_gmm_v0', 'hybrid', 'hierarchical']:
        # Add magnitude/energy-like features
        features['magnitude'] = np.sqrt((features ** 2).sum(axis=1))
        
        # Add ratios for all adjacent pairs (common in physics)
        col_names = features_df.columns
        for i in range(len(col_names) - 1):
            col1, col2 = col_names[i], col_names[i+1]
            features[f'{col2}_to_{col1}_ratio'] = np.log1p(features_df[col2]) - np.log1p(features_df[col1])
        
        # Add log-transformed features for exponential-like data
        for col in features_df.columns:
            features[f'log_{col}'] = np.log1p(features_df[col])
            
        # Add squared features (energy-like)
        for col in features_df.columns:
            features[f'sq_{col}'] = features_df[col] ** 2
    
    return features

def apply_specialized_scaling(features, scaler_type):
    """Apply scaling optimized for physics data with exponential distribution"""
    
    if scaler_type == 'quantile':
        # QuantileTransformer works well for non-normal distributions
        scaler = QuantileTransformer(n_quantiles=min(1000, len(features)//2), 
                                   output_distribution='uniform', random_state=42)
    elif scaler_type == 'power':
        # PowerTransformer for making data more Gaussian
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    try:
        scaled_features = scaler.fit_transform(features)
        return scaled_features
    except:
        # Fallback to robust scaling if specialized scaling fails
        print("Falling back to RobustScaler")
        return RobustScaler().fit_transform(features)

def detect_and_weight_outliers(features):
    """Detect outliers and create weights"""
    outlier_detector = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = outlier_detector.fit_predict(features)
    
    # Create weights: lower weight for outliers
    weights = np.ones(len(features))
    weights[outlier_labels == -1] = 0.5
    
    print(f"Detected {(outlier_labels == -1).sum()} outliers ({(outlier_labels == -1).mean():.1%})")
    return weights

def focus_on_adjacent_dims(scaled_features, column_names, adjacent_focus):
    """Focus on specified adjacent dimensions"""
    
    # Extract dimension indices from focus string (e.g., 'dim23' -> indices 1,2)
    dim_numbers = [int(c) for c in adjacent_focus if c.isdigit()]
    if len(dim_numbers) >= 2:
        dim1_idx = dim_numbers[0] - 1  # Convert to 0-indexed
        dim2_idx = dim_numbers[1] - 1
        
        # Ensure indices are valid
        max_idx = min(len(column_names) - 1, scaled_features.shape[1] - 1)
        dim1_idx = min(dim1_idx, max_idx)
        dim2_idx = min(dim2_idx, max_idx)
        
        print(f"Focusing on dimensions {dim1_idx+1} and {dim2_idx+1}")
        
        # Create focused feature set
        n_original_dims = len(column_names)
        focused_features = scaled_features[:, :n_original_dims].copy()
        
        # Emphasize the specified adjacent dimensions
        focused_features[:, dim1_idx] *= 3.0  # Strong emphasis
        focused_features[:, dim2_idx] *= 3.0  # Strong emphasis
        
        # Add interaction between these dimensions
        interaction_features = np.column_stack([
            scaled_features[:, dim1_idx] * scaled_features[:, dim2_idx],  # Product
            np.abs(scaled_features[:, dim1_idx] - scaled_features[:, dim2_idx]),  # Abs diff
            (scaled_features[:, dim1_idx] + scaled_features[:, dim2_idx]) / 2  # Average
        ])
        
        focused_features = np.column_stack([focused_features, interaction_features])
        return focused_features
    else:
        return scaled_features

def create_hybrid_features(scaled_features, column_names, adjacent_focus):
    """Create hybrid feature representation with multiple adjacent pairs"""
    hybrid_features = scaled_features.copy()
    
    n_dims = len(column_names)
    
    # Add interactions for all adjacent pairs
    interaction_features = []
    for i in range(n_dims - 1):
        dim1_vals = scaled_features[:, i]
        dim2_vals = scaled_features[:, i + 1]
        
        # Weight based on which pair we're focusing on
        weight = 1.0
        if adjacent_focus and f"dim{i+1}{i+2}" == adjacent_focus:
            weight = 2.0  # Give extra weight to the focused pair
        
        pair_interactions = np.column_stack([
            dim1_vals * dim2_vals * weight,           # Product
            np.abs(dim1_vals - dim2_vals) * weight,   # Absolute difference
            (dim1_vals + dim2_vals) * weight / 2      # Average
        ])
        interaction_features.append(pair_interactions)
    
    # Combine all interaction features
    if interaction_features:
        all_interactions = np.column_stack(interaction_features)
        hybrid_features = np.column_stack([hybrid_features, all_interactions])
    
    return hybrid_features

def create_specialized_gmm(k_clusters, covariance_type, features):
    """Create GMM specifically optimized for physics data with exponential distribution"""
    
    # Analyze feature characteristics to determine best GMM parameters
    n_samples, n_features = features.shape
    
    # For exponential-like data, use more conservative regularization
    reg_covar = 1e-4 if np.any(np.var(features, axis=0) < 0.1) else 1e-6
    
    # Adjust covariance type for physics data characteristics
    if covariance_type == 'full' and n_features > n_samples // 10:
        # Too many parameters for full covariance, use tied instead
        covariance_type = 'tied'
        print(f"Adjusted covariance type to '{covariance_type}' for better numerical stability")
    
    # Use more initialization attempts for exponential-like data
    n_init = min(50, max(20, 100 // k_clusters))
    
    # Create specialized GMM
    gmm = GaussianMixture(
        n_components=k_clusters,
        covariance_type=covariance_type,
        init_params='k-means++',  # Better initialization for skewed data
        n_init=n_init,
        max_iter=2000,  # More iterations for convergence
        tol=1e-7,  # Stricter tolerance
        reg_covar=reg_covar,  # Adaptive regularization
        random_state=42,
        warm_start=False,
        verbose=0
    )

    # Pre-process: remove extreme outliers that might hurt GMM fitting
    robust_scaler = RobustScaler()
    features_robust = robust_scaler.fit_transform(features)
    
    # Detect extreme outliers (beyond 3 MAD)
    median_abs_dev = np.median(np.abs(features_robust - np.median(features_robust, axis=0)), axis=0)
    outlier_threshold = 3.0
    extreme_outliers = np.any(np.abs(features_robust) > outlier_threshold, axis=1)
    
    if extreme_outliers.sum() > 0:
        print(f"Detected {extreme_outliers.sum()} extreme outliers for GMM preprocessing")
        # Fit on non-extreme data, then predict on all data
        clean_features = features[~extreme_outliers]
        if len(clean_features) > k_clusters * 10:  # Ensure enough data
            print("Fitting GMM on cleaned data")
            # Create a wrapper that fits on clean data but predicts on all data
            class CleanFitGMM:
                def __init__(self, base_gmm, clean_mask, all_features):
                    self.base_gmm = base_gmm
                    self.clean_mask = clean_mask
                    self.all_features = all_features
                    self.fitted = False
                
                def fit(self, X):
                    clean_X = X[~self.clean_mask]
                    self.base_gmm.fit(clean_X)
                    self.fitted = True
                    return self
                
                def predict(self, X):
                    if not self.fitted:
                        raise ValueError("Model not fitted yet")
                    return self.base_gmm.predict(X)
                
                def fit_predict(self, X):
                    self.fit(X)
                    return self.predict(X)
            
            return CleanFitGMM(gmm, extreme_outliers, features)
    
    return gmm

def create_specialized_gmm_v0(k_clusters, covariance_type, features):
    """Create GMM optimized for physics data"""
    return GaussianMixture(
        n_components=k_clusters,
        covariance_type=covariance_type,
        init_params='kmeans',
        n_init=30,  # More initializations
        max_iter=1000,  # More iterations
        tol=1e-6,  # Stricter tolerance
        reg_covar=1e-6,  # Regularization for numerical stability
        random_state=42
    )

def create_adjacent_focused_model(k_clusters, covariance_type):
    """Create model focused on adjacent dimension structure"""
    # Use GMM with tied covariance (good for similar cluster shapes in physics)
    return GaussianMixture(
        n_components=k_clusters,
        covariance_type=covariance_type,  # tied, Physics events often have similar covariance structure
        init_params='k-means++',
        n_init=60,  # More attempts for focused features
        max_iter=1500,
        tol=1e-7,
        reg_covar=1e-5,  # Slightly more regularization for focused features
        random_state=42
    )

def create_hybrid_model(k_clusters, covariance_type):
    """Create hybrid model that considers outlier weights and adjacent structure"""
    # For now, use standard GMM (sample_weight not available in sklearn GMM)
    # Use full covariance for complex relationships in hybrid features
    return GaussianMixture(
        n_components=k_clusters,
        covariance_type=covariance_type,  # More flexible for complex hybrid relationships
        init_params='k-means++',
        n_init=40,
        max_iter=1200,
        tol=1e-6,
        reg_covar=1e-5,  # Balance between stability and flexibility
        random_state=42
    )

def create_gmm_model(k_clusters, covariance_type):
    return GaussianMixture(n_components=k_clusters,
        covariance_type=covariance_type,
        init_params='kmeans',
        n_init=20,
        max_iter=500,
        tol=1e-5,
        random_state=42
    )

def create_kmeans_model(k_clusters):
    return KMeans(n_clusters=k_clusters,
        init='k-means++',
        n_init=20,
        max_iter=500,
        random_state=42
    )

def perform_hierarchical_clustering(k_clusters, adjacent_focus, features):
    """
    Perform two-level hierarchical clustering to obtain exactly k_clusters clusters.
    First level uses DBSCAN to separate major clusters based on selected dimensions.
    Second level uses GMM to refine each major cluster.
    """
    if adjacent_focus is None:
        print("`adjacent_focus` not provided, defaulting to dimensions [1, 2] (Dim2 and Dim3).")
        dim_indices = [1, 2]
    else:
        try:
            dims_str = re.findall(r'[1-9]', adjacent_focus)
            if len(dims_str) < 2:
                raise ValueError("Please provide at least two dimension digits, e.g., 'dim23' for Dim2 and Dim3.")
            dim_indices = [int(d) - 1 for d in dims_str[:2]]
            print(f"First-level clustering will focus on dimensions: {dim_indices}")
            
        except (TypeError, ValueError) as e:
            print(f"Error parsing adjacent_focus ('{adjacent_focus}'). Falling back to KMeans. Error: {e}")
            
            # Fallback: Use KMeans to ensure exactly k_clusters clusters
            print(f"Fallback strategy: Using KMeans to directly form {k_clusters} clusters")
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(features)
    
    if max(dim_indices) >= features.shape[1]:
        print("Warning: Specified dimension index is out of range. Using fallback strategy - KMeans.")
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    first_level_features = features[:, dim_indices]
    
    # First-level clustering: Separate major clusters
    print("Performing first-level clustering (major cluster separation)...")
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    main_clusters = dbscan.fit_predict(first_level_features)
    
    # Refine DBSCAN labels (assign noise points to nearest cluster)
    unique_clusters = np.unique(main_clusters[main_clusters != -1])
    if -1 in main_clusters:
        noise_mask = (main_clusters == -1)
        core_points = first_level_features[~noise_mask]
        core_labels = main_clusters[~noise_mask]
        if len(core_points) > 0:
            for i in np.where(noise_mask)[0]:
                dists = distance.cdist([first_level_features[i]], core_points)
                closest_idx = np.argmin(dists)
                main_clusters[i] = core_labels[closest_idx]
        else:
            print("Warning: DBSCAN found no core points. Using fallback strategy - KMeans.")
            n_fallback_clusters = max(2, min(k_clusters // 2, len(features) // 10))
            kmeans = KMeans(n_clusters=n_fallback_clusters, random_state=42, n_init=10)
            main_clusters = kmeans.fit_predict(first_level_features)
    
    n_main_clusters = len(np.unique(main_clusters))
    if n_main_clusters == 0:
        print("Error: No major clusters formed. Falling back to KMeans.")
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    print(f"Identified {n_main_clusters} major clusters.")
    
    # Second-level clustering: Refine each major cluster
    print("Performing second-level clustering (refinement within major clusters)...")
    
    cluster_sizes = []
    cluster_indices = []
    for cluster_idx in range(n_main_clusters):
        cluster_mask = (main_clusters == cluster_idx)
        cluster_size = np.sum(cluster_mask)
        cluster_sizes.append(cluster_size)
        cluster_indices.append(cluster_idx)
    
    subclusters_per_main = distribute_subclusters(cluster_sizes, k_clusters)
    
    final_labels = np.zeros(len(features), dtype=int)
    current_label = 0
    actual_clusters_created = 0
    
    for cluster_idx, n_subclusters in zip(cluster_indices, subclusters_per_main):
        cluster_mask = (main_clusters == cluster_idx)
        cluster_data = features[cluster_mask]
        
        if len(cluster_data) == 0:
            continue
        
        if n_subclusters <= 1 or len(cluster_data) < n_subclusters:
            final_labels[cluster_mask] = current_label
            actual_clusters_created += 1
            current_label += 1
        else:
            try:
                gmm = GaussianMixture(
                    n_components=n_subclusters, 
                    covariance_type='tied',
                    n_init=3, 
                    random_state=42,
                    max_iter=100
                )
                sub_labels = gmm.fit_predict(cluster_data)
                
                unique_sub_labels = np.unique(sub_labels)
                if len(unique_sub_labels) != n_subclusters:
                    # GMM did not produce the expected number of clusters. Using fallback strategy - KMeans.
                    kmeans_sub = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                    sub_labels = kmeans_sub.fit_predict(cluster_data)
                
                final_labels[cluster_mask] = current_label + sub_labels
                actual_clusters_created += n_subclusters
                current_label += n_subclusters
                
            except Exception as e:
                print(f"GMM failed on cluster {cluster_idx}: {e}. Using fallback strategy - KMeans.")
                kmeans_sub = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                sub_labels = kmeans_sub.fit_predict(cluster_data)
                final_labels[cluster_mask] = current_label + sub_labels
                actual_clusters_created += n_subclusters
                current_label += n_subclusters
    
    unique_labels = np.unique(final_labels)
    n_actual_clusters = len(unique_labels)
    
    print(f"Actual clusters produced: {n_actual_clusters}, Target: {k_clusters}")
    
    if n_actual_clusters < k_clusters:
        print(f"Warning: Fewer clusters ({n_actual_clusters}) than target ({k_clusters}). Adjusting...")
        final_labels = adjust_to_target_clusters(features, final_labels, k_clusters)
    elif n_actual_clusters > k_clusters:
        print(f"Warning: More clusters ({n_actual_clusters}) than target ({k_clusters}). Merging...")
        final_labels = merge_excess_clusters(final_labels, k_clusters)
    
    # Ensure consecutive labeling
    final_labels = pd.factorize(final_labels)[0]
    
    print(f"Final number of clusters: {len(np.unique(final_labels))}")
    return final_labels

def distribute_subclusters(cluster_sizes, k_clusters):
    """
    Distribute subclusters among major clusters proportionally to reach k_clusters.
    """
    n_main_clusters = len(cluster_sizes)
    total_samples = sum(cluster_sizes)
    
    # Basic allocation: each main cluster gets at least one subcluster
    subclusters_allocation = [1] * n_main_clusters
    remaining_subclusters = k_clusters - n_main_clusters
    
    if remaining_subclusters > 0:
        # Distribute remaining subclusters proportionally based on cluster sizes
        cluster_proportions = [size / total_samples for size in cluster_sizes]
        
        # Calculate how many extra subclusters each cluster should get
        extra_subclusters = [int(remaining_subclusters * prop) for prop in cluster_proportions]
        
        # Handle rounding remainder
        assigned_extra = sum(extra_subclusters)
        remaining_extra = remaining_subclusters - assigned_extra
        
        # Assign remaining subclusters to the largest clusters
        if remaining_extra > 0:
            sorted_indices = sorted(range(n_main_clusters), key=lambda i: cluster_sizes[i], reverse=True)
            for i in range(remaining_extra):
                extra_subclusters[sorted_indices[i]] += 1
        
        # Update final allocation
        for i in range(n_main_clusters):
            subclusters_allocation[i] += extra_subclusters[i]
    
    return subclusters_allocation

def adjust_to_target_clusters(features, labels, k_clusters):
    """
    When number of clusters is less than target, split the largest cluster further.
    """
    current_k = len(np.unique(labels))
    needed_clusters = k_clusters - current_k
    
    while needed_clusters > 0:
        # Find the largest cluster to split
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster_idx = unique_labels[np.argmax(counts)]
        largest_cluster_mask = (labels == largest_cluster_idx)
        largest_cluster_data = features[largest_cluster_mask]
        
        if len(largest_cluster_data) < 2:
            # Cannot split if cluster has fewer than 2 samples
            break
            
        # Split the largest cluster into 2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(largest_cluster_data)
        
        # Update labels
        max_label = np.max(labels)
        labels[largest_cluster_mask] = np.where(sub_labels == 0, largest_cluster_idx, max_label + 1)
        
        needed_clusters -= 1
    
    return labels

def merge_excess_clusters(labels, k_clusters):
    """
    When number of clusters exceeds target, merge the smallest clusters together.
    """
    while len(np.unique(labels)) > k_clusters:
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Identify the two smallest clusters to merge
        sorted_indices = np.argsort(counts)
        smallest_label = unique_labels[sorted_indices[0]]
        second_smallest_label = unique_labels[sorted_indices[1]]
        
        # Merge the smallest cluster into the second smallest
        labels[labels == smallest_label] = second_smallest_label
    
    return labels

def refine_clusters_physics_aware(cluster_labels, original_features):
    """Post-process clusters based on physics intuition"""
    # Physics-based refinement: events with similar total energy should be in similar clusters
    total_energy = np.sum(original_features.values, axis=1)
    
    # Simple refinement: ensure no cluster is too small
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    min_cluster_size = max(5, len(cluster_labels) // (len(unique_labels) * 20))  # At least 5% of average
    
    for label, count in zip(unique_labels, counts):
        if count < min_cluster_size:
            # Reassign small clusters based on energy similarity
            mask = cluster_labels == label
            if mask.sum() > 0:
                small_cluster_energies = total_energy[mask]
                
                # Find most similar cluster based on energy distribution
                best_target_label = label
                min_energy_diff = float('inf')
                
                for other_label in unique_labels:
                    if other_label != label and np.sum(cluster_labels == other_label) >= min_cluster_size:
                        other_energies = total_energy[cluster_labels == other_label]
                        # Compare median energies
                        energy_diff = abs(np.median(small_cluster_energies) - np.median(other_energies))
                        if energy_diff < min_energy_diff:
                            min_energy_diff = energy_diff
                            best_target_label = other_label
                
                if best_target_label != label:
                    print(f"Reassigning small cluster {label} ({count} points) to cluster {best_target_label} based on energy similarity")
                    cluster_labels[mask] = best_target_label
    
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Perform clustering on a dataset.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file (e.g., public_data.csv, private_data.csv)")
    parser.add_argument("output_file", type=str, help="Path to save the clustered CSV file (e.g., public_submission.csv, b11902091_public.csv, private_submission.csv)")
    parser.add_argument("--method", type=str, choices=['kmeans', 'hierarchical', 'gmm', 'specialized_gmm', 'specialized_gmm_v0', 'adjacent_focus', 'hybrid'], default='gmm',
                        help="Clustering method to use (default: 'gmm'). Note: 'hierarchical' and 'adjacent_focus' require --adjacent_focus parameter")
    parser.add_argument("--scaler", type=str, choices=['standard', 'robust', 'quantile', 'power'], default='standard',
                        help="Scaler type to use: 'standard', 'robust', 'quantile', or 'power' (default: 'standard')")
    parser.add_argument("--covariance", type=str, choices=['full', 'tied', 'diag', 'spherical'], default='tied',
                        help="Covariance type for GMM methods: 'full', 'tied', 'diag', or 'spherical' (default: 'tied'). Note: Only applies to 'gmm', 'specialized_gmm', 'specialized_gmm_v0', 'adjacent_focus', and 'hybrid' methods. For 'kmeans' and 'hierarchical', this parameter is ignored")
    parser.add_argument("--adjacent_focus", type=str, default=None,
                        help="Adjacent focus dimension pair (e.g., 'dim12', 'dim23', 'dim34', ...) (default: 'None'). REQUIRED for 'hierarchical', 'adjacent_focus' methods. Optional for 'hybrid' method. Not used for other methods")
    parser.add_argument("--feature", action='store_true',
                        help="Enable specialized feature engineering for physics data (default: disabled). This will create additional features based on physics principles, such as energy ratios and magnitudes")
    parser.add_argument("--outlier", action='store_true',
                        help="Enable outlier detection and handling (default: disabled). This will mark outliers and potentially weight them differently during clustering")
    parser.add_argument("--refine", action='store_true',
                        help="Enable physics-based refinement of clusters (default: disabled). This will adjust clusters based on physics intuition, such as ensuring similar energy events are grouped together")

    args = parser.parse_args()
    
    # Validate arguments based on method requirements
    if args.method in ['hierarchical', 'adjacent_focus'] and args.adjacent_focus is None:
        parser.error(f"Method '{args.method}' requires --adjacent_focus parameter (e.g., 'dim12', 'dim23', etc.)")
    
    # Handle covariance parameter for non-GMM methods
    covariance_type = args.covariance if args.method in ['gmm', 'specialized_gmm', 'specialized_gmm_v0', 'adjacent_focus', 'hybrid'] else None
    
    print(f"Dataset selected: {args.input_file}")
    print(f"Method: {args.method}, Scaler: {args.scaler}, Covariance: {covariance_type or 'None'}, Adjacent Focus: {args.adjacent_focus or 'None'}")
    
    if args.method in ['kmeans', 'hierarchical'] and args.covariance != 'full':
        print(f"Note: Covariance parameter is ignored for method '{args.method}'")
    
    flags = []
    if args.feature:
        flags += ["feature"]
    if args.outlier:
        flags += ["outlier"]
    if args.refine:
        flags += ["refine"]

    perform_clustering(
        args.input_file, 
        args.output_file,
        method=args.method,
        scaler_type=args.scaler,
        covariance_type=covariance_type,
        adjacent_focus=args.adjacent_focus,
        flags=flags
    )
    
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()