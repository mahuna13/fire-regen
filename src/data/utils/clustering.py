from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


def cluster(df, feature_cols, n_clusters=5, verbose=0):
    # Limit threads, without it BLAS causes segmentation error.
    # Error documentation: https://github.com/xianyi/OpenBLAS/issues/3180
    threadpool_limits(limits=1)
    threadpool_limits(limits=1, user_api='blas')

    km = MiniBatchKMeans(n_clusters=n_clusters, verbose=verbose, n_init='auto')
    km.fit(df[feature_cols])
    output = df.assign(cluster=km.predict(df[feature_cols]))

    return output
