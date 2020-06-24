import os
import logging
from timeit import default_timer as timer
import re
import sys
import subprocess

from genomic_tools_lib import Logging
from genomic_tools_lib.file_formats import Parquet

from genomic_tools_lib.external_tools.tensorqtl import genotypeio, trans
from pyarrow import parquet as pq
import pandas
import numpy


# def main(args):
#     """
#     1. load phenotype and covariate data
#     2. transform, write, and gzip BED file
#     3. load BED and genotype
#     :param args:
#     :return:
#     """
#     # Ensure env is set up correctly:
#     single_chrom_files = helpers.check_if_formattable(args.geno)
#     if single_chrom_files:
#         files = helpers.format_chrom_file_names(args.geno)
#     else:
#         files = [args.geno]
#     for file in files:
#         helpers.ensure_files_in_place(file)
#     helpers.check_cuda()
#     helpers.make_dir(args.out)
#
#     # covar_df = helpers.load_pheno_covar_df(args.covar)
#     pheno_df = helpers.load_pheno_covar_df(args.pheno)
#     pheno_df.index = pheno_df.index.astype('str')
#     intersection_index, covar_df = merge_data_sources(pheno_df.index, args.covar, files[0])
#     pheno_df = pheno_df.reindex(intersection_index).T
#     n_phenos = helpers.find_n_phenos(args.pheno)
#
#     trans_df = pd.DataFrame(columns = ['variant_id', 'phenotype_id', 'pval', 'maf'])
#
#     for file in files:
#         plink_reader = tensorqtl.genotypeio.PlinkReader(file,
#                                                         select_samples = [str(i) for i in intersection_index],
#                                                         verbose = False)
#         geno_df = plink_reader.load_genotypes()
#         logging.info("Loaded genotype data from {}".format(file))
#         trans_df_i = tensorqtl.trans.map_trans(geno_df,
#                                                pheno_df,
#                                                covar_df,
#                                                batch_size = 2000,
#                                                pval_threshold = args.pval_threshold,
#                                                maf_threshold = args.maf_threshold,
#                                                verbose = False)
#         trans_df = trans_df.append(trans_df_i)
#         logging.info("Finished analysis for genotype file {}".format(file))
#
#     # Write logs and write output
#     logging.info('Finished with QTL analysis')
#     n_matched_phenos = len(trans_df['phenotype_id'].unique())
#     if n_matched_phenos < n_phenos:
#         s = "Only {n_matched_phenos} out of {n_phenos} phenotypes showed significant associations." \
#             " The p-value threshold may be too low".format(n_matched_phenos = n_matched_phenos,
#                                                            n_phenos = n_phenos)
#         logging.info(s)
#     trans_results_fp = os.path.join(args.out, TODAY + '_trans-qtl-results.txt')
#     trans_df.to_csv(trans_results_fp, sep = '\t', index = False)

class FileOut:
    def __init__(self, out_fp, gzip=False):
        pass


class FileIn:
    def __init__(self, plink_geno, parquet_pheno):
        self.BIM = ".bim"
        self.BED = ".bed"
        self.FAM = ".fam"
        self.geno_pre = plink_geno
        self.pheno_fp = parquet_pheno
        self.pheno = pq.ParquetFile(parquet_pheno)
        self.covar_df = None

        self.individuals = self._find_intersection()

    def get_pheno(self): return self._load_pheno(self.individuals)
    def get_covar(self): return self.covar_df

    def get_geno(self):
        logging.info("Loading genotype")
        pr = genotypeio.PlinkReader(self.geno_pre, verbose=False)
        genotype_df = pr.load_genotypes()
        genotype_df.columns = [str(i) for i in genotype_df.columns]
        logging.info("Finished loading genotype")
#        print(genotype_df.head())
#        print(genotype_df.shape)
        return genotype_df

    def _find_intersection(self):
        fam_df = pandas.read_csv(self.geno_pre + self.FAM,
                             sep='\s',
                             header=None,
                             names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'],
                             usecols=['IID'],
                             engine='python')
        geno_ids = set(fam_df.IID.astype(str).values)
        logging.log(9, "Individuals from genotype: {}".format(len(geno_ids)))

        pheno_ids = set(Parquet._read(self.pheno, columns=['individual'])['individual'])
        logging.log(9, "Individuals from phenotype: {}".format(len(pheno_ids)))

        # I have so much trouble remembering python set methods.
        # This is intersection.
        all_ids = geno_ids.intersection(pheno_ids)
        logging.info("Working with {} individuals".format(len(all_ids)))
        return all_ids

    def _load_pheno(self, individuals):
        """
        What tensorqtl wants: pandas DataFrame with pheno names as rows and
            individual IDs as columns
        """
        df = Parquet._read(self.pheno, to_pandas=True,
                           specific_individuals=individuals)
        logging.info("Loaded {} phenotypes".format(df.shape[1]))
        df['individual'] = df['individual'].astype('str')
        df = df.set_index('individual')
        self.covar_df = pandas.DataFrame(index=df.index)
#        print(df.shape)
        # ind_index = pd.Index(individuals)
        # df = df.reindex(ind_index)
        # df_ind = df['individual']
        # df = df.drop('individual', axis=1)
        return df.T

def run(args):
    """
    1. Read plink and parquet files and find intersection of individuals.
    2. Load parquet phenotypes and get into correct data structure.
    3. Do some type of batching?
    4. run tensorqtl
    5. format results
    6. write results
    :param args:
    :return:
    """
    start = timer()
    #logging.info("Using tensorqtl version {}".format(tensorqtl.__version__))
    file_in = FileIn(args.plink_genotype,
                     args.parquet_phenotype)
    pheno_df = file_in.get_pheno()
    covar_df = file_in.get_covar()
    geno_df = file_in.get_geno()
    ss_df = trans.map_trans(geno_df,
                                       pheno_df,
                                       covar_df,
                                       batch_size=2000,
                                       pval_threshold=1e-5,
                                       maf_threshold=0.05,
                                       verbose=False)
    print(ss_df.head())
    print(ss_df.columns)
    print(ss_df.shape)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Fast GWAS summary statistics from tensorQTL")

    parser.add_argument("-plink_genotype")
    # parser.add_argument("-plink_genotype_pattern")
    parser.add_argument("-parquet_phenotype")
    parser.add_argument("-output_dir")
    parser.add_argument('--parsimony', default=10, type=int)

    args = parser.parse_args()

    Logging.configure_logging(level=args.parsimony, target=sys.stdout)

    run(args)
