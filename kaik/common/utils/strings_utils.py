def separated(values, *, limit, stringify, sep):
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f"{sep}... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = sep.join(stringify(x) for x in values)
    return rendered + continuation


def comma_sep(values, limit=20, stringify=repr):
    return separated(values, limit=limit, stringify=stringify, sep=", ")
