import parser
import parsed_queries


if __name__ == "__main__":
    import importlib
    importlib.reload(parser)
    importlib.reload(parsed_queries)

    q_list = ["f=234&xtype=lower&f=54", "f=42&xtype=high", "f=53&allow-deprecated=true&xtype=lower"]
    parsed = [parser.to_dict(q) for q in q_list]
    pq = parsed_queries.ParsedQueries(parsed)
