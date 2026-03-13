""" strip_comments.py

Helper script to remove comments from python scripts recursively to generate
a clean repo

"""

import argparse
from pathlib import Path
import io
import tokenize

def remove_comments_and_docstrings(input_file):
    """
    Returns 'source' minus comments and docstrings.

    Taken from stackoverflow:
        https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
    """
    with open(input_file, "r") as f:
        source = f.read()

    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
        # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out

def get_args():
    """ get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Name of directory to strip of comments (in place)",)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    rec_dir = Path(args.dir)
    for py_file in rec_dir.rglob("*.py"):
        stripped_file = remove_comments_and_docstrings(py_file)
        with open(py_file, "w") as fp:
            fp.write(stripped_file)
