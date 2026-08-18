def clean_sentence_pairs(df):
    """
    Remove sentence pairs with conflicting labels and
    keep only one occurrence of consistent duplicate pairs.
    """

    pair_columns = ["sentence1", "sentence2"]

    # Find sentence pairs that have more than one label
    conflicting_pairs = (
        df.groupby(pair_columns)["label"]
        .nunique()
        .reset_index()
    )

    conflicting_pairs = conflicting_pairs[
        conflicting_pairs["label"] > 1
    ]

    # Remove all rows that belong to conflicting pairs
    clean_df = df.merge(
        conflicting_pairs[pair_columns],
        on=pair_columns,
        how="left",
        indicator=True
    )

    clean_df = clean_df[
        clean_df["_merge"] == "left_only"
    ].drop(columns="_merge")

    # Keep only one occurrence of the remaining duplicate pairs
    clean_df = clean_df.drop_duplicates(
        subset=pair_columns
    ).reset_index(drop=True)

    return clean_df
