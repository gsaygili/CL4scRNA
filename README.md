# CL4scRNA

# Continual Learning Approaches for Single Cell RNA Sequencing Data
## Gorkem Saygili and Busra Ozgode Yigin
### Cognitive Sciences and Artificial Intelligence, Tilburg School of Humanities and Digital Sciences, Tilburg University, The Netherlands.

Abstract:
Single-cell RNA sequencing data is among the most interesting and impactful data of today and the sizes of the available datasets are increasing drastically. There is a substantial need for learning from large datasets, causing nontrivial challenges, especially in hardware. Loading even a single dataset into the memory of an ordinary, off-the-shelf computer can be infeasible, and using computing servers might not always be an option. This paper presents continual learning as a solution to such hardware bottlenecks. The findings of cell-type classification demonstrate that XGBoost and Catboost algorithms, when implemented in a continual learning framework, exhibit superior performance compared to the best-performing static classifier. We achieved up to 10% higher median F1 scores than the state-of-the-art on the most challenging datasets. On the other hand, these algorithms can suffer from variations in data characteristics across diverse datasets, pointing out indications of the catastrophic forgetting problem.

*** [1] G. Saygili and B. Ozgode Yigin, "Continual learning approaches for single cell RNA sequencing data", Nature Scientific Reports, 2023.***

*** Please cite our paper [1] in case you use the code.***

Created by Busra Ozgode Yigin and Gorkem Saygili on 14-09-23.

Datasets can be obtained from: 
- Intra and Inter Datasets by [abdelaal et al., 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1795-z): https://doi.org/10.5281/zenodo.3357167
- PBMC-eQTL~\cite{michielsen2021hierarchical}: https://zenodo.org/record/3736493#.ZGZgXHZBxPY 
- HLCA latent space ~\cite{michielsen2022single}: https://zenodo.org/record/6337966#.YqmGIidBx3g

***Important Note: This code is under MIT License:***

***THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.***

