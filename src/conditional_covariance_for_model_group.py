__author__ = "alvaro barbeira"

import logging
import os
import sqlite3
import pandas
import numpy
import gzip


from genomic_tools_lib import Logging, Utilities
from genomic_tools_lib.data_management import TextFileTools
from genomic_tools_lib.miscellaneous import matrices, Genomics
from genomic_tools_lib.file_formats import Parquet

from covariance_for_model import n_, get_file_map

def run(args):

    logging.info("Loading group")
    groups = pandas.read_table(args.group)
    groups = groups.assign(chromosome = groups.gtex_intron_id.str.split(":").str.get(0))
    groups = groups.assign(position=groups.gtex_intron_id.str.split(":").str.get(1))
    groups = Genomics.sort(groups)

    logging.info("Getting parquet genotypes")
    file_map = get_file_map(args)

    logging.info("Getting genes")
    with sqlite3.connect(args.model_db_group_key) as connection:
        # Pay heed to the order. This avoids arbitrariness in sqlite3 loading of results.
        extra = pandas.read_sql("SELECT * FROM EXTRA order by gene", connection)
        extra = extra[extra["n.snps.in.model"] > 0]

    individuals = TextFileTools.load_list(args.individuals) if args.individuals else None

    logging.info("Getting list of wanted genes")
    want_genes = pandas.read_csv(args.want_genes, header=None)[0].values.tolist()

    logging.info("Processing")
    Utilities.ensure_requisite_folders(args.output_dir)

    genes_ = groups[["chromosome", "position", "gene_id"]].drop_duplicates()
    cond_tracker = [] # Track condition values of sigma_11 by gene
    with sqlite3.connect(args.model_db_group_key) as db_group_key:
        with sqlite3.connect(args.model_db_group_values) as db_group_values:
            for i,t_ in enumerate(genes_.itertuples()):
                g_ = t_.gene_id
                if g_ not in want_genes:
                    continue
                ofile = os.path.join(args.output_dir, "{}__{}.txt.gz".format(g_, args.tissue))
                if os.path.exists(ofile):
                    logging.info("Output already exists, either delete it or move it")
                    continue
                with gzip.open(ofile, "w") as f:
                    f.write("GENE RSID1 RSID2 VALUE\n".encode())
                    chr_ = t_.chromosome.split("chr")[1]
                    logging.log(8, "Proccessing %i/%i:%s", i+1, len(genes_), g_)

                    if not n_.search(chr_):
                        logging.log(9, "Unsupported chromosome: %s", chr_)
                        continue
                    dosage = file_map[int(chr_)]

                    group = groups[groups.gene_id == g_]
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

                    cond_tracker.append((g_, sigma_11_cond))
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
                    for value in group.intron_id.unique():
                        c_write = matrices._flatten_matrix_data([(value, pred_ids, covar_cond)])
                        for entry in c_write:
                            l = "{} {} {} {}\n".format(entry[0], entry[1], entry[2], entry[3])
                            f.write(l.encode())
    logging.info("Finished building covariance.")
    cond_tracker = pandas.DataFrame(cond_tracker, columns=["gene", "sigma_11_cond"])

    cond_out = os.path.join(args.output_dir, "{}.sigma11.condition.numbers.tsv".format(args.tissue))
    cond_tracker.to_csv(cond_out, index=False, sep="\t")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate BSLMM runs on study")
    parser.add_argument("-parquet_genotype_folder", help="Parquet Genotype folder")
    parser.add_argument("-parquet_genotype_pattern", help="Pattern to detect parquet genotypes by chromosome")
    parser.add_argument("-model_db_group_key", help="Model file with group keys as genes")
    parser.add_argument("-model_db_group_values", help="Model file with group values as genes")
    parser.add_argument("-group", help="group definitions")
    parser.add_argument("-output_dir", help="Where to save stuff")
    parser.add_argument("--output_rsids", action="store_true")
    parser.add_argument("--individuals")
    parser.add_argument("-parsimony", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default = "10")

    # Arguments to support conditional covariance calculation for a single gene
    parser.add_argument("-want_genes", help="What genes to focus on for.")
    parser.add_argument("-condition_info_dir", help="Directory containing files that specify SNPs used for prediction of introns for want_genes.")
    parser.add_argument("-pred_pattern", help="Python format string pattern of file (no header) in condition_info_dir that contains SNPs used for predicting the introns. E.g. '{want_genes}.snps.for.intron.pred.tsv")
    parser.add_argument("-cond_pattern", help="Python format string pattern of file (no header) in condition_info_dir that contains SNPs used for conditioning in COJO. E.g. '{want_genes}.snps.for.intron.pred.tsv")
    parser.add_argument("-tissue", help="Name of tissue (should match individuals being given).")
    parser.add_argument("-rcond", help="Cutoff for small singular values in numpy.linalg.pinv", type=float, default=1e-15)

    args = parser.parse_args()

    Logging.configure_logging(int(args.parsimony))

    run(args)
