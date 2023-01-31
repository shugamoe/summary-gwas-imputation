__author__ = "alvaro barbeira"

import logging
import os
import re
from datetime import datetime
import sqlite3
import pandas
import numpy
import gzip


from metax.misc import KeyedDataSource
from genomic_tools_lib import Logging, Utilities
from genomic_tools_lib.miscellaneous import matrices, PandasHelpers

def get_file_map(args, want_chr=None):
    from pyarrow import parquet as pq
    logging.log(9, "Loading parquet files")
    r = re.compile(args.parquet_genotype_pattern)
    files = os.listdir(args.parquet_genotype_folder)
    files = {int(r.search(f).groups()[0]):os.path.join(args.parquet_genotype_folder, f) for f in files if r.search(f)}
    p = {}
    keys = sorted(files.keys())
    for k in files.keys():
        v = files[k]
        if want_chr != None:
            if want_chr not in v:
                continue
        logging.log(9, "Loading %i:%s", k, v)
        g = pq.ParquetFile(v)
        p[k] = g
    return p

n_ = re.compile("^(\d+)$")

def run(args):
    if os.path.exists(args.output):
        logging.info("Output already exists, either delete it or move it")
        return


    logging.info("Getting genes")
    with sqlite3.connect(args.model_db) as connection:
        # Pay heed to the order. This avoids arbitrariness in sqlite3 loading of results.
        extra = pandas.read_sql("SELECT DISTINCT gene FROM EXTRA WHERE 'n.snps.in.model' > 0 order by gene", connection)

    if args.covar_mode:
        from genomic_tools_lib.data_management import TextFileTools
        from genomic_tools_lib.file_formats import Parquet
        logging.info("Getting parquet genotypes")
        file_map = get_file_map(args)
        individuals = TextFileTools.load_list(args.individuals) if args.individuals else None
        logging.info("Getting list of wanted genes")
        want_genes = pandas.read_csv(args.want_genes, header=None)[0].values.tolist()

    logging.info("Processing")
    Utilities.ensure_requisite_folders(args.output)

    if args.gene_whitelist:
        gene_whitelist = pandas.read_csv(args.gene_whitelist[0], usecols=[args.gene_whitelist[1]])[args.gene_whitelist[1]].to_list()

    # Imputed SNPs don't have effect sizes or standard errors, but they
    # recorded the position window of the original SNPs, with effect sizes,
    # that were used to impute their zscores, go get those
    if args.get_og_for_imputed:
        logging.info("Loading COJO MA File | {}".format(datetime.now()))
        cojo_ma_df = pandas.read_csv(args.get_og_for_imputed, sep=" ", usecols=['chromosome_position', 'effect_size'])
        cojo_ma_df['position'] = cojo_ma_df['chromosome_position'].str.split("_", expand=True)[1].astype(int)
        cojo_ma_df['chr_num'] = cojo_ma_df["chromosome_position"].str.extract(r"(\d{1,2})").astype(int)

        logging.info("Loading Imputation Ranges | {}".format(datetime.now()))
        snp_typed_imp_pos = KeyedDataSource.load_data_dual(args.gwas_file, "panel_variant_id", "typed_min_pos", 
                             "typed_max_pos", sep="\t", numeric=True) 
        if args.simple_variants:
            snp_typed_imp_pos = {"{}_{}".format(vkey.split("_")[0],
                vkey.split("_")[1]): snp_typed_imp_pos[vkey] for vkey in snp_typed_imp_pos}
        logging.info("Imputation Ranges Loaded | {}".format(datetime.now()))

    results = []
    with sqlite3.connect(args.model_db) as connection:
        for i,t in enumerate(extra.itertuples()):
            g_ = t.gene

            # Beautiful. . . 
            if args.gene_whitelist:
                if g_ not in gene_whitelist:
                    continue
            if args.want_genes:
                if g_ not in want_genes:
                    continue

            logging.log(9, "Proccessing %i/%i:%s", i+1, extra.shape[0], g_)
            w = pandas.read_sql("select * from weights where gene = '{}';".format(g_), connection)
            if "chr" in w.varID.values[0]:
                chr_ = w.varID.values[0].split("_")[0].split("chr")[1]
            else:
                chr_ = w.varID.values[0].split("_")[0].split("_")[0]

            if not n_.search(chr_):
                logging.log(9, "Unsupported chromosome: %s", chr_)
                continue

            if args.covar_mode:
                ofile = os.path.join(args.output_dir, "{}__{}.txt.gz".format(g_, args.tissue))
                if os.path.exists(ofile):
                    logging.info("Output already exists, either delete it or move it")
                    continue
                with gzip.open(ofile, "w") as f:
                    f.write("GENE RSID1 RSID2 VALUE\n".encode())
                    dosage = file_map[int(chr_)]

                    pred_snps = \
                    pandas.read_csv(os.path.join(args.condition_info_dir,
                        args.pred_pattern.format(want_gene=g_)),
                        header=None)

                    cond_snps = \
                    pandas.read_csv(os.path.join(args.condition_info_dir,
                        args.cond_pattern.format(want_gene=g_)),
                        header=None)

                    all_snps = pandas.concat([cond_snps, pred_snps])
                    if individuals:
                        d = Parquet._read(dosage, columns=all_snps.iloc[:,0].values, specific_individuals=individuals)
                        del d["individual"]
                    else:
                        d = Parquet._read(dosage, columns=all_snps.iloc[:,0].values, skip_individuals=True)

                    var_ids = list(d.keys())
                    if len(var_ids) != len(all_snps):
                        logging.info("Some SNPs missing, determining which prediction and conditional SNPs are left.")
                        survive = pandas.DataFrame({0: var_ids})
                        cond_snps = cond_snps.merge(survive)
                        pred_snps = pred_snps.merge(survive)
                        all_snps = pandas.concat([cond_snps, pred_snps])

                    var_ids = all_snps.iloc[:,0].values
                    pred_ids = pred_snps.iloc[:, 0].values
                    num_pred = len(pred_snps)
                    num_cond = len(cond_snps)
                    num_tot = num_cond + num_pred

                    c = numpy.cov([d[x] for x in var_ids])

                    # https://stats.stackexchange.com/questions/186402/expressing-conditional-covariance-matrix-in-terms-of-covariance-matrix
                    sigma_11 = c[0:num_cond, 0:num_cond]
                    sigma_11_cond = numpy.linalg.cond(sigma_11)
                    sigma_12 = c[0:num_cond, num_cond:num_tot]
                    sigma_21 = c[num_cond:num_tot, 0:num_cond]
                    sigma_22 = c[num_cond:num_tot, num_cond:num_tot]

                    results.append((g_, sigma_11_cond))
                    try:
                        covar_cond = sigma_22 - numpy.dot(numpy.dot(sigma_21, numpy.linalg.pinv(sigma_11)), sigma_12)
                    except numpy.linalg.linalg.LinAlgError as e:
                        logging.info("Singular Matrix probably for {}?".format(g_))
                        print(e)
                        logging.info("Using pseudoinverse for {}".format(g_))

                        try:
                            sigma_11_pinv = numpy.linalg.pinv(sigma_11, rcond=args.rcond)
                            covar_cond = sigma_22 - numpy.dot(numpy.dot(sigma_21, sigma_11_pinv), sigma_12)
                        except numpy.linalg.linalg.LinAlgError as e:
                            logging.info("Pseudo-inverse failure for {}?".format(g_))
                            print(e)
                            continue

                    if covar_cond.shape != (num_pred, num_pred):
                        raise Exception("Shape of conditional covariance matrix doesn't match number of prediction SNPs")
                    # This will produce redundant data, but if we're limiting to one
                    # gene's introns only it shouldn't be too bad.

                    c_write = matrices._flatten_matrix_data([(g_, pred_ids, covar_cond)])
                    for entry in c_write:
                        l = "{} {} {} {}\n".format(entry[0], entry[1], entry[2], entry[3])
                        f.write(l.encode())
            else:
                try:
                    ofile = None
                    variants_ = sorted(w.varID.values)

                    variants_no_tiss_split_ = [variant.split("_") for variant in variants_]
                    variants_chr_pos_ = ["{}_{}".format(split[0], split[1]) if "chr" in split[0] else
                    "chr{}_{}".format(split[0], split[1]) for split in variants_no_tiss_split_]
                            
                    gene_df = pandas.DataFrame({"chromosome_position": variants_chr_pos_})
                    gene_df = gene_df.drop_duplicates() # Remove SNPs that were used across tissues

                    all_pred_ofile = os.path.join(args.output_dir, "{}.snplist.all.for.prediction".format(g_))
                    gene_df[['chromosome_position']].to_csv(all_pred_ofile, header=False, index=False)

                    if args.get_og_for_imputed:
                # gene_df_imp_snps gives us the 'imputed' SNPs that we don't
                # have original data for (effect_size, standard error) COJO
                # needs this information to run, so instead, we go through each
                # gene and get the position windows for the original SNPs (that
                # have effect_size, standard error)
                        gene_df_imp_snps = cojo_ma_df[cojo_ma_df['effect_size'].isna()].merge(gene_df, on="chromosome_position")
                        gene_df_og_snps = cojo_ma_df.dropna().merge(gene_df, on="chromosome_position")
                        ofile = os.path.join(args.output_dir, "{}.snplist".format(g_))
                        gene_df_og_snps[['chromosome_position']].to_csv(ofile, header=False, index=False)
                        results.append((g_, ofile))
                        continue
                        
                        if len(gene_df_imp_snps) == 0: # All SNPs for the introns for a gene are original SNPs
                            ofile = os.path.join(args.output_dir, "{}.snplist".format(g_))
                            gene_df_og_snps[['chromosome_position']].to_csv(ofile, header=False, index=False)
                            results.append((g_, ofile))
                            continue
                        
                # Retrieve the original SNPs in the windows of original SNPs
                # used to impute the imputed SNPS
                        var_chr_num =  int(gene_df_imp_snps['chr_num'][0])
                        
                        og_snps_for_imp_df = None 
                        for imp_snp in gene_df_imp_snps['chromosome_position'].tolist():
                            full_name_index = variants_chr_pos_.index(imp_snp)
                            if args.simple_variants:
                                full_name = imp_snp
                            else:
                                full_name = variants_[full_name_index]

                            min_og_snp_pos = snp_typed_imp_pos[full_name]["typed_min_pos"]
                            max_og_snp_pos = snp_typed_imp_pos[full_name]["typed_max_pos"]
                            
                            og_snps_for_imp = cojo_ma_df[cojo_ma_df['chr_num'] == var_chr_num].dropna()
                            og_snps_for_imp = og_snps_for_imp[og_snps_for_imp['position'].between(min_og_snp_pos, max_og_snp_pos)]
                            
                            if og_snps_for_imp_df is None:
                                og_snps_for_imp_df = og_snps_for_imp
                            else:
                                og_snps_for_imp_df = pandas.concat([og_snps_for_imp_df, og_snps_for_imp])
                                og_snps_for_imp_df = og_snps_for_imp_df.drop_duplicates() # Windows might overlap for different SNPs
                # Combine original SNPs from original variant list and original
                # SNPs needed for imputation of imputed snps
                        all_og_snp_df = pandas.concat([og_snps_for_imp_df[['chromosome_position']], 
                                                      gene_df_og_snps[['chromosome_position']]]) 
                        all_og_snp_df = all_og_snp_df.drop_duplicates()
                        ofile = os.path.join(args.output_dir, "{}.snplist".format(g_))
                        all_og_snp_df[['chromosome_position']].to_csv(ofile, header=False, index=False)
                    else: 
                        ofile = os.path.join(args.output_dir, "{}.snplist".format(g_))
                        gene_df_og_snps[['chromosome_position']].to_csv(ofile, header=False, index=False)
                
                except Exception as e:
                    #TODO: improve error tracking
                    logging.log(8, "{} Exception: {}".format(g_, str(e)))
                    status = "Error"
                    # results.append((g_, ofile))
                else:
                    results.append((g_, ofile))
    if args.covar_mode:
        logging.info("Finished building covariance.")
        out_cols = ["gene", "sigma_11_cond"]
    else:
        logging.info("Finished creating gene expression groups")
        out_cols = ["group", "snp_list_path"]

    results = pandas.DataFrame(results, columns=out_cols)

    logging.info("Saving metadata")
    Utilities.save_dataframe(results, args.output)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate BSLMM runs on study")
    parser.add_argument("--parquet_genotype_folder", help="Parquet Genotype folder")
    parser.add_argument("--parquet_genotype_pattern", help="Pattern to detect parquet genotypes by chromosome")
    parser.add_argument("--model_db", help="Where to save stuff")
    parser.add_argument("--output", help="Where to save stuff")
    parser.add_argument("--output_rsids", action="store_true")
    parser.add_argument("--simple_variants", help="Use for PWAS to match variants.", action="store_true")
    parser.add_argument("--individuals")
    parser.add_argument("--parsimony", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default = "10")

    # Args to support creating groups for COJO
    parser.add_argument("--gwas_file", help="Load a single GWAS file. (Alternative to providing a gwas_folder and gwas_file_pattern)")
    parser.add_argument("--get_og_for_imputed", help="For SNPs that were imputed, write the SNP list of the original genes used to impute those SNPs. Input an .ma file for COJO, any row without na will be considered an 'original' gene.")
    parser.add_argument("--gene_whitelist", nargs="+", default = [], help="Path to tab separated file with genes to run, followed by the column name with ENSG* style gene names.")

    # Args to support conditioned covariance calculations
    parser.add_argument("--covar_mode", default=False, action="store_true", help="Path to tab separated file with genes to run, followed by the column name with ENSG* style gene names.")
    parser.add_argument("--output_dir", help="Where to save stuff")
    parser.add_argument("--want_genes", help="What genes to focus on for.")
    parser.add_argument("--condition_info_dir", help="Directory containing files that specify SNPs used for prediction of introns for want_genes.")
    parser.add_argument("--pred_pattern", help="Python format string pattern of file (no header) in condition_info_dir that contains SNPs used for predicting the introns. E.g. '{want_genes}.snps.for.intron.pred.tsv")
    parser.add_argument("--cond_pattern", help="Python format string pattern of file (no header) in condition_info_dir that contains SNPs used for conditioning in COJO. E.g. '{want_genes}.snps.for.intron.pred.tsv")
    parser.add_argument("--tissue", help="Name of tissue (should match individuals being given).")
    parser.add_argument("--rcond", help="Cutoff for small singular values in numpy.linalg.pinv", type=float, default=1e-15)

    args = parser.parse_args()

    Logging.configure_logging(int(args.parsimony))

    run(args)
