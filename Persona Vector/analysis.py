import pandas as pd
import torch
import os
import numpy as np
from create_vector import *
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import textwrap

def svd_analysis(matrix):
    """Args: vector (torch.Tensor): A 2D tensor where each row represents a difference vector."""

    # print(f"\nFirst 10 elements of vector:")
    # if matrix.dim() == 2:
    #     print(f"Layer 0 first 10 elements: {matrix[0, :10]}")
    #     if matrix.shape[0] > 1:
    #         print(f"Layer 1 first 10 elements: {matrix[1, :10]}")
    # else:
    #     print(f"First 10 elements: {matrix[:10]}")
    
    # # Check for NaN or Inf values
    # nan_count = torch.isnan(matrix).sum().item()
    # inf_count = torch.isinf(matrix).sum().item()
    # print(f"\nNaN values: {nan_count}")
    # print(f"Inf values: {inf_count}")
    # print(f"Total elements: {matrix.numel()}")
    
    # # Print vector information
    # print(f"\nVector type: {type(matrix)}")
    # print(f"Vector shape: {matrix.shape}")
    # print(f"Vector dimensions: {matrix.dim()}")
    # print(f"Vector dtype: {matrix.dtype}")
    # print(f"Vector device: {matrix.device}")
    
    # # Additional analysis
    # if matrix.dim() == 2:
    #     print(f"Number of layers: {matrix.shape[0]}")
    #     print(f"Hidden dimension: {matrix.shape[1]}")
    # elif matrix.dim() == 1:
    #     print(f"Vector length: {matrix.shape[0]}")
    
    # Statistics
    print(f"\nVector statistics:")
    print(f"Min value: {matrix.min().item():.6f}")
    print(f"Max value: {matrix.max().item():.6f}")
    print(f"Mean value: {matrix.mean().item():.6f}")
    print(f"Std deviation: {matrix.std().item():.6f}")
    print(f"L2 norm: {torch.norm(matrix).item():.6f}")

    # set_trace()
    print(f"\nSVD Analysis:")
    
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    print(f"SVD shapes - U: {U.shape}, S: {S.shape}, V^T: {Vt.shape}")
    print(f"Number of singular values: {len(S)}")
    print(f"Top 10 singular values: {S[:10].tolist()}")
    print(f"Largest singular value: {S[0].item():.6f}")
    print(f"Smallest singular value: {S[-1].item():.6f}")
    
    # Condition number (ratio of largest to smallest singular value)
    if S[-1] > 1e-12:  # Avoid division by very small numbers
        condition_number = S[0] / S[-1]
        print(f"Condition number: {condition_number.item():.6f}")
    else:
        print(f"Condition number: inf (singular matrix)")
    
    # set_trace()
    # Rank estimation (count non-zero singular values with tolerance)
    tolerance = 1e-6
    rank = (S > tolerance).sum().item()
    print(f"Estimated rank (tol={tolerance}): {rank}")


    # Cumulative energy (explained variance ratio)
    energy = S.pow(2)
    cumulative_energy = torch.cumsum(energy, dim=0) / energy.sum()
    print(f"Energy in top 1 component: {cumulative_energy[0].item():.4f}")
    print(f"Energy in top 5 components: {cumulative_energy[4].item():.4f}")
    print(f"Energy in top 10 components: {cumulative_energy[9].item():.4f}")


    print('One layer finished')
    # plt.plot(cumulative_energy.numpy())
    # plt.xlabel("Number of components")
    # plt.ylabel("Cumulative energy (explained variance)")
    # plt.title("Cumulative Energy Spectrum")
    # plt.grid(True)
    # plt.savefig("cumulative_energy_spectrum.png")


def conduct_svd_and_projection(X_2d, singular_ls):
    # print(f"Matrix shape for SVD: {X_2d.shape}")
    
    # Step 1: Conduct SVD decomposition
    print("Performing SVD: X = U @ S @ V^T")
    U, S, Vt = torch.linalg.svd(X_2d, full_matrices=False)
    
    # print(f"  U shape: {U.shape}")
    # print(f"  S shape: {S.shape}")  
    # print(f"  V^T shape: {Vt.shape}")
    # print(f"  Dominant singular value (σ₁): {S[0]:.6f}")

    value_ls, score_ls = [], []
    # Step 2: Project X onto the corresponding dominant vector (σ₁)
    for singular_index in singular_ls:
        v1 = Vt[singular_index]  # Dominant right singular vector
        sigma1 = S[singular_index]  # Dominant singular value

        # print(f"Dominant vector u₁ shape: {v1.shape}")
        # print(f"Dominant singular value σ₁: {sigma1:.6f}")
        # print(f"Top 10 singular values: {S[:10].tolist()}")
        # print(f"Largest singular value: {S[0].item():.6f}")
        # print(f"Smallest singular value: {S[-1].item():.6f}")

        # Project each row of X onto v1
        scores = X_2d @ v1
        
        # print(f"Projection scores shape: {scores.shape}")
        # print(f"Score statistics:")
        # print(f"  Mean: {torch.mean(scores):.6f}")
        # print(f"  Std: {torch.std(scores):.6f}")
        # print(f"  Min: {torch.min(scores):.6f}")
        # print(f"  Max: {torch.max(scores):.6f}")
        value_ls.append(sigma1.item())
        score_ls.append(scores.cpu().numpy())

    res = {'dominant singular value list': value_ls, 'projection scores list': score_ls}

    # u1 = U[:, 0]
    # print(f"First left singular vector u₁ statistics:")
    # print(f"  Mean: {torch.mean(u1):.6f}")
    # print(f"  Std: {torch.std(u1):.6f}")
    # print(f"  Min: {torch.min(u1):.6f}")
    # print(f"  Max: {torch.max(u1):.6f}")


    return res

def find_correlation(projected_scores, gt_scores):
    print("\n=== Calculating Score Correlation ===")
    

    # Create arrays with valid (non-None) ratings
    valid_indices = []
    valid_gt_scores = []
    valid_scores = []
    
    for i in range(len(gt_scores)):
        if (gt_scores[i] is not None):
            valid_indices.append(i)
            valid_gt_scores.append(gt_scores[i])
            valid_scores.append(projected_scores[i])

    if len(valid_indices) == 0:
        print("No valid rating pairs found for correlation analysis!")
        return {}
    
    valid_gt_scores = np.array(valid_gt_scores)
    valid_scores = np.array(valid_scores)


    # Calculate MAE
    # mae = np.mean(np.abs(valid_scores - valid_gt_scores))

    # proj = valid_scores
    # proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    # mae = np.mean(np.abs(proj_norm - 0))
    # print(f"Mean Absolute Error (MAE): {mae:.6f}")

    # Calculate correlations
    corr_gt = np.corrcoef(valid_scores, valid_gt_scores)[0, 1]
    print(f"Correlation with ground truth ratings: {corr_gt:.6f}")

    return corr_gt

# TODO: Do TSNE plot to see cluster patterns
def tsne_plot(diff_vector, num_clusters=10, dir_name='tsne_plots', labels=None):
    num_layers = diff_vector.shape[0]
    
    for layer in tqdm(range(num_layers)):
        # Cluster using kmeans to get color labels
        kmeans = KMeans(n_clusters=num_clusters)
        color_labels = kmeans.fit_predict(diff_vector[layer].numpy())

        tsne = TSNE(n_components=2, random_state=0)
        response_avg_diff_2d = tsne.fit_transform(diff_vector[layer].numpy())
        # plt.figure(figsize=(8, 8))
        # plt.scatter(response_avg_diff_2d[:, 0], response_avg_diff_2d[:, 1], c=color_labels, cmap='tab10')
        # plt.title(f'TSNE of Response Diff - Layer {layer}')
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.grid(True)
        # filepath = f"{dir_name}_{num_clusters}/response_diff_layer_{layer}.png"
        # plt.savefig(filepath, bbox_inches='tight')
        # plt.close()

        # Interactive plot with plotly with hover labels
        if labels:
            fig = px.scatter(x=response_avg_diff_2d[:, 0], y=response_avg_diff_2d[:, 1], color=color_labels.astype(str),
                            title=f'PCA of Response Diff - Layer {layer}',
                            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'color': 'Cluster'},
                            hover_name=labels[0],
                            hover_data={'Code': labels[1], 'x': False, 'y': False, 'color': False})

            fig.update_layout(hoverlabel=dict(font_size=8))
            interactive_filepath = os.path.join(dir_name, f'plotly/response_diff_layer_{layer}.html')
            fig.write_html(interactive_filepath)


def pca_plot(diff_vector, num_clusters=10, dir_name='pca_plots', n_component=2, labels=None):
    num_layers = diff_vector.shape[0]
    for layer in tqdm(range(num_layers)):
        # Cluster using kmeans to get color labels
        kmeans = KMeans(n_clusters=num_clusters)
        color_labels = kmeans.fit_predict(diff_vector[layer].numpy())

        pca = PCA(n_components=n_component)
        response_avg_diff_2d = pca.fit_transform(diff_vector[layer].numpy())
        # plt.figure(figsize=(8, 8))
        # plt.scatter(response_avg_diff_2d[:, 0], response_avg_diff_2d[:, 1], c=color_labels, cmap='tab10')
        # plt.title(f'PCA of Response Diff - Layer {layer}')
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.grid(True)
        # filepath = f"{dir_name}/response_diff_layer_{layer}.png"
        # plt.savefig(filepath, bbox_inches='tight')
        # plt.close()

        # Interactive plot with plotly with hover labels
        if labels:
            x_axs = f"PC{n_component-1}"
            y_axs = f"PC{n_component}"
            df = pd.DataFrame({
                x_axs: response_avg_diff_2d[:, n_component-2],
                y_axs: response_avg_diff_2d[:, n_component-1],
                'Cluster': color_labels.astype(str),
                'Problem': labels[0],
                'Code': labels[1],
                'KC': labels[2],
                'Error': labels[3]
            })


            fig = px.scatter(df, x=x_axs, y=y_axs, color='Cluster',
                title=f'PCA of Response Diff - Layer {layer}',
                labels={x_axs: f'Principal Component {n_component-1}', y_axs: f'Principal Component {n_component}', 'Cluster': 'Cluster'},
                hover_data={'Problem': True,'Code': True, 'KC': True, 'Error': True, x_axs: False, y_axs: False, 'Cluster': False}
            )

            fig.update_layout(hoverlabel=dict(font_size=13))
            interactive_filepath = os.path.join(dir_name, f'plotly{n_component-1}{n_component}/response_diff_layer_{layer}.html')
            fig.write_html(interactive_filepath)

        # # Get elbow plot for PCA explained variance ratio
        # pca_full = PCA(n_components=diff_vector.shape[2])
        # pca_full.fit(diff_vector[layer].numpy())
        # plt.figure(figsize=(8, 5))


        # plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1), pca_full.explained_variance_ratio_, marker='o')
        # plt.title(f'PCA Explained Variance Ratio - Layer {layer}')
        # plt.xlabel('Principal Component')
        # plt.ylabel('Variance Ratio')
        # plt.grid(True)
        # elbow_filepath = os.path.join(dir_name, f'elbow/pca_explained_variance_layer_{layer}.png')
        # plt.savefig(elbow_filepath, bbox_inches='tight')
        # plt.close()


def main():
    vector_diff = torch.load('response_avg_diff_inc_v1.pt', weights_only=True)

    # print(f"Loaded vector_diff shape: {vector_diff.shape}")
    file_paths = {
        "problem_ls": "problem_list_all_inc_v1.json",
        "code_ls": "code_list_all_inc_v1.json",
        "problem_kc_map": "baseline_kc_mapping.json",
        "error_cnt_ls": "error_cnt_all_inc.json",
        "error_types_ls": "error_types_all_inc.json",
    }

    data = {}
    for key, path in file_paths.items():
        with open(path, "r") as f:
            data[key] = json.load(f)

    # Access them as:
    problem_ls = data["problem_ls"]
    code_ls = data["code_ls"]
    problem_kc_map = data["problem_kc_map"]
    error_cnt_ls = data["error_cnt_ls"]
    error_types_ls = data["error_types_ls"]

    # with open('solution_gpt4o.json', 'r') as f:
    #     solution = json.load(f)

    #     for key, val in solution.items():
    #         print(key)
    #         print(val)
    #         print('-----------------------------')

    # set_trace()
    # Plot PCA for all layers
    err_summary_ls = [f"{cnt} errors: {error_ls}" for cnt, error_ls in zip(error_cnt_ls, error_types_ls)]

    kc_map = [sorted(problem_kc_map[problem]) for problem in problem_ls]

    problem_proc_ls = ['\n'.join(textwrap.wrap(problem, width=100)) for problem in problem_ls]
    problem_fin_ls = [problem_i.replace("\n", "<br>") for problem_i in problem_proc_ls]
    problem_fin_ls = [problem_i+"<br>" for problem_i in problem_fin_ls]

    code_ls_mod = [code_i.replace("\n", "<br>") for code_i in code_ls]
    code_ls_mod = [code_i+"<br>" for code_i in code_ls_mod]
    pca_plot(vector_diff, num_clusters=10, dir_name='pca_plots_all_inc', n_component=16, labels=[problem_fin_ls, code_ls_mod, kc_map, err_summary_ls])
    print('Done')

    # # Analyze each layer on correlation
    # device = torch.device('cuda')
    # vector_diff = vector_diff.to(device)
    # with open('code_len_diff_inc.json', 'r') as f:
    #     gt_scores = json.load(f)

    #     layers = list(range(33))
    #     res_ls = []

    #     top_singular_values = 20
    #     singular_ls = list(range(top_singular_values))
    #     res_nested_ls = [[] for _ in range(top_singular_values)]


    #     for layer in layers:
    #         matrix_i = vector_diff[layer, :, :] 

    #         print(f"Layer {layer} analysis:")
    #         svd_analysis(matrix_i)

    #         results = conduct_svd_and_projection(matrix_i, singular_ls)

    #         projected_scores_ls = results['projection scores list']

    #         for idx, projected_scores in enumerate(projected_scores_ls):
    #             res = find_correlation(projected_scores, gt_scores)
    #             res_nested_ls[idx].append(res)
                

    #     fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    #     num_lines = len(res_nested_ls)
    #     colors = plt.cm.tab20(np.linspace(0, 1, num_lines))


    #     for k, color in enumerate(colors):
    #         ax.plot(layers, res_nested_ls[k], label=f'K={k}', color=color)

    #     ax.set_title('Correlation with Length Diff by Layer')
    #     ax.set_xlabel('Layer')
    #     ax.set_ylabel('Correlation')
    #     ax.set_ylim(-1, 1)
    #     ax.grid(True, alpha=0.3)

    #     # compact legend outside on the right
    #     ax.legend(
    #         title='Singular Val',
    #         loc='center left', bbox_to_anchor=(1.02, 0.5),  # just outside the axes
    #         facecolor='white', framealpha=1, edgecolor='black',
    #         fontsize=8, title_fontsize=9, markerscale=0.8,
    #         handlelength=1.2, labelspacing=0.3, handletextpad=0.4, borderpad=0.3
    #     )

    #     file_name = f"v2/corr_len_diff_by_layer_singular_values_{top_singular_values}.png"
    #     # save without the big white strip on the right
    #     fig.savefig(file_name, dpi=200,
    #                 bbox_inches='tight', pad_inches=0.05)

    #     print("Finished plotting correlation by layer.")


main()