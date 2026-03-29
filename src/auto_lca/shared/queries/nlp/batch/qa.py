QA_INPUT_QUERY = """select distinct pid, abstract, request_id
From ist-lca.scrape.paper
where abstract is not null and @__SUBQUERY__"""  # TODO
