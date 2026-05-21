"""
PCA Baseline

Simple PCA dimensionality reduction as a lower bound baseline.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
import anndata as ad


class PCABaseline:
    """
    PCA baseline for dimensionality reduction
    
    Args:
        n_components: Number of principal components
        random_state: Random seed
    """
    
    def __init__(
        self,
        n_components: int = 50,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit PCA on data
        
        Args:
            X: Input data (n_samples, n_features)
        """
        self.pca.fit(X)
        self.fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise RuntimeError("PCA must be fitted before transform")
        
        return self.pca.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def fit_transform_adata(
        self,
        adata: ad.AnnData,
        output_key: str = 'X_pca'
    ) -> ad.AnnData:
        """
        Fit and transform AnnData object
        
        Args:
            adata: Input AnnData
            output_key: Key to store results
            
        Returns:
            AnnData with PCA embeddings
        """
        # Get data
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Fit and transform
        embeddings = self.fit_transform(X)
        
        # Store results
        adata.obsm[output_key] = embeddings
        
        # Store explained variance
        adata.uns[f'{output_key}_variance_ratio'] = self.pca.explained_variance_ratio_
        
        print(f"PCA complete: {self.n_components} components")
        print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        return adata
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio"""
        if not self.fitted:
            raise RuntimeError("PCA must be fitted first")
        return self.pca.explained_variance_ratio_
    
    @property
    def components(self) -> np.ndarray:
        """Get principal components"""
        if not self.fitted:
            raise RuntimeError("PCA must be fitted first")
        return self.pca.components_

