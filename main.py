import argparse

import faker

from query_parsing import parsed_queries, parser
import file_io
import graphix
import fake_query_generator.fq_generator as fq_g
import fake_query_generator.fq_cluster as fq_c
import fake_query_generator.fq_param as fq_p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="File name")
    ap.add_argument("-c", "--column", help="Column name to be used for parsing when the given file is a CSV file.")
    ap.add_argument("-s", "--skip-header", help="The number of skip row when the given file is a TXT file.", action='store_true')

    args = ap.parse_args()

    file_name: str = args.file
    if file_name is None:
        print("Use random sample data")

        fake = faker.Faker()
        fq_gen = fq_g.FQGenerator(
            [
                fq_c.FQCluster([fq_p.FQParam("name1", fake.name)], 0.4),
                fq_c.FQCluster([fq_p.FQParam("name2", fake.name)], 0.6)
            ]
        )
        urls = fq_gen.generate(40)
    else:
        if file_name.endswith("csv"):
            assert args.column is not None, "Input column name"
            urls = file_io.read_csv(file_name, args.column, ",")
        elif file_name.endswith("tsv"):
            assert args.column is not None, "Input column name"
            urls = file_io.read_csv(file_name, args.column, "\t")
        elif file_name.endswith("txt"):
            urls = file_io.read_txt(file_name, bool(args.skip_header))
        else:
            raise ValueError(f"Not supported file type: {file_name}")

    parsed = parser.to_dict(urls)
    pq = parsed_queries.ParsedQueries(parsed)

    dist = pq.get_total_dist()
    graphix.draw_dist_mat(dist)

    coord = dist.reconstruct_coord()
    graphix.draw_coord(coord)

    errors = coord.calc_reconstruction_error()
    graphix.draw_reconstruction_error(errors)
