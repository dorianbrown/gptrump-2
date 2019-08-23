import textwrap


def print_messages(text_col, linewidth=80):
    for msg in text_col:
        print("-"*linewidth)
        print(textwrap.fill(msg, linewidth))
    print("-" * 80)
    pass
