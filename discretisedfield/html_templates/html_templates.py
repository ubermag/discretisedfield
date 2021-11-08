import jinja2
import os


def html_template(name):
    """Return html template with the given name."""
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')))
    return env.get_template(name + '.jinja2')
