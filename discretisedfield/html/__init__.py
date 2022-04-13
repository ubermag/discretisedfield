import os
import re

import jinja2


def get_template(name):
    """Return html template with the given name."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        )
    )
    return env.get_template(name + ".jinja2")


def strip_tags(string):
    """Strip all html tags and convert lists."""
    string = re.sub(r"\s+", "", string)
    string = re.sub(r"<ul>", "(", string)
    string = re.sub(r"<li>", "", string)
    string = re.sub(r"</li></ul>", ")", string)
    string = re.sub(r"</li>", ",", string)
    string = re.sub(r"</?i>", "`", string)  # subregion names are italic
    string = re.sub(r"<[^<]+>", "", string)
    string = re.sub(r"([,:])", r"\1 ", string)
    return string
