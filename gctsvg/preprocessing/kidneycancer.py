def extract_sc_patient_data(adata, patient_id):
    adata = adata[adata.obs['patient'] == patient_id].copy()
    return adata

def st_samples_in_name(sample_ids, name):
    return any(sample_id in name for sample_id in sample_ids)

def st_samples_not_in_name(sample_ids, name):
    return all(sample_id not in name for sample_id in sample_ids)

def extract_st_patient_data(adata, patient_id):
    adata = adata[adata.obs['donor_id'] == patient_id].copy()
    
    target_sample_ids = adata[adata.obs['donor_id'] == patient_id].obs['sample'].unique().tolist()
    columns_to_keep = ['feature_types', 'genome', 'SYMBOL', 'mt']
 
    columns_to_keep.extend([colname for colname in adata.var.columns if st_samples_in_name(target_sample_ids, colname)])
    
    columns_to_drop = [col for col in adata.var.columns if col not in columns_to_keep]
    adata.var.drop(columns=columns_to_drop, inplace=True)

    libraries_to_remove = [key for key in adata.uns['spatial'].keys() if st_samples_not_in_name(target_sample_ids, key)]
    
    for library_key in libraries_to_remove:
        adata.uns['spatial'].pop(library_key, None)
        
    return adata

