class Utils:
    """Utilities specific to the GTEx pipeline."""

    @staticmethod
    def get_index_of_string_in_list_case_insensitive(item: str, lst: list[str]) -> int:
        """Get the index of a string in a list in a case-insensitive manner.

        Parameters
        ----------
        item: str
            The item to look for.
        list: list[str]
            The list in which to look for the item.

        Returns
        -------
        idx: int
            The index of the item in the list if it exists; -1 if not.
        """
        for i, x in enumerate(lst):
            if item.casefold() == x.casefold():
                return i
        return -1
