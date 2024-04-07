import argparse

import parser
import parsed_queries
import io
import url_locater
import gen_sample_data
import graphix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="File name")
    ap.add_argument("-c", "--column", help="Column name to be used for parsing when the given file is a CSV file.")
    ap.add_argument("-s", "--skip-header", help="The number of skip row when the given file is a TXT file.", action='store_true')

    args = ap.parse_args()

    file_name: str = args.file
    if file_name is None:
        print("Use random sample data")
        ori_coord, dist = gen_sample_data.gen_data(10)
        coord = url_locater.locate(dist)
        graphix.draw_coord_for_sample(coord, ori_coord)
    else:
        if file_name.endswith("csv"):
            assert args.column is not None, "Input column name"
            urls = io.read_csv(file_name, args.column, ",")
        elif file_name.endswith("tsv"):
            assert args.column is not None, "Input column name"
            urls = io.read_csv(file_name, args.column, "\t")
        elif file_name.endswith("txt"):
            urls = io.read_txt(file_name, bool(args.skip_header))
        else:
            raise ValueError(f"Not supported file type: {file_name}")

        parsed = parser.to_dict(urls)
        pq = parsed_queries.ParsedQueries(parsed)

        dist = pq.get_total_dist()
        graphix.draw_dist_mat(dist)

        coord = url_locater.locate(dist)
        graphix.draw_coord(coord)