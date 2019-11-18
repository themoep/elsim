# This file is part of Elsim
#
# Copyright (C) 2019, Sebastian Bachmann <hello at reox.at>
# All rights reserved.
#
# Elsim is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Elsim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Elsim.  If not, see <http://www.gnu.org/licenses/>.
import os

import click

from elsim import ELSIM_VERSION, Elsim, Eldiff
from elsim.similarity import Compress
from elsim.dalvik import (
        ProxyDalvikString,
        FILTERS_DALVIK_SIM_STRING,
        ProxyDalvik,
        FILTERS_DALVIK_SIM,
        ProxyDalvikMethod,
        FILTERS_DALVIK_BB,
        ProxyDalvikBasicBlock,
        FILTERS_DALVIK_DIFF_BB,
        DiffDalvikMethod,
        )
import elsim
from elsim.utils import load_analysis

def check_one_file(dx1, dx2, FS, threshold, compressor, details, view_strings, new, deleted, diff, score):
    """
    Show similarities between two dalvik containers

    :param androguard.core.analysis.analysis.Analysis dx1: first file
    :param androguard.core.analysis.analysis.Analysis dx2: second file
    :param dict FS: the filter basis
    :param str compressor: compressor name
    :param bool details: should extra information be shown
    :param bool view_strings: also calculate the similarities based on strings
    :param bool new: should the similarity score include new elements
    :param bool deleted: should the similarity score include deleted elements
    :param bool diff: display the difference
    :param bool score: only print the score
    """
    #from pyinstrument import Profiler
    #profiler = Profiler()
    #profiler.start()

    el = Elsim(ProxyDalvik(dx1), ProxyDalvik(dx2), FS, threshold, compressor)
    if score:
        click.echo("Methods: {:7.4f}".format(el.get_similarity_value(new, deleted)))
    else:
        print("Calculating similarity based on methods")
        el.show(new, deleted, details)

    if view_strings:
        els = Elsim(ProxyDalvikString(dx1), ProxyDalvikString(dx2), FILTERS_DALVIK_SIM_STRING, threshold, compressor)
        if score:
            click.echo("Strings: {:7.4f}".format(els.get_similarity_value(new, deleted)))
        else:
            print("Calculating similarity based on strings")
            els.show(new, deleted, details)

    if diff and not score:
        for i, j in el.split_elements():
            # split_elements returns tuples of similar elements
            # Get a list if "Method" objects
            # Instead of using the similarity on the whole Method, we calculate the similarites between the basic blocks
            # FIXME: having like thousand classes here seems to be overcomplicated...
            elb = Elsim(ProxyDalvikMethod(i), ProxyDalvikMethod(j), FILTERS_DALVIK_BB, threshold, compressor)
            eld = Eldiff(ProxyDalvikBasicBlock(elb), FILTERS_DALVIK_DIFF_BB)
            ddm = DiffDalvikMethod(i, j, elb, eld)
            ddm.show()

    #profiler.stop()
    #print(profiler.output_text(unicode=True, color=True))


@click.command()
@click.version_option(ELSIM_VERSION)
@click.option("-d", "--details", is_flag=True, help="display detailed information about the changes")
@click.option("--diff", is_flag=True, help="Show the difference between the files. Does not work if --score is used.")
@click.option("-c", "--compressor", default="BZ2", type=click.Choice([x.name for x in Compress]),
        show_default=True,
        show_choices=True,
        help="Set the compression method. Some methods perform better,"
        " as they can compress the content much better, but usually"
        " come with the drawback, that they are very slow. "
        "While LZMA has the best compression, Snappy has well compression but"
        " is much faster than LZMA.")
@click.option("-t", "--threshold", default=0.6, type=click.FloatRange(0, 1),
        help="Threshold when sorting interesting items")
@click.option("-s", "--size", type=int,
        help='exclude specific method below the specific size (specify the minimum size of a method to be used (it is the length (bytes) of the dalvik method)')
@click.option("-e", "--exclude", type=str, help="exlude class names (python regex string)")
@click.option("--new/--no-new", help="calculate similarity score by including new elements", show_default=True)
@click.option("--deleted/--no-deleted", help="calculate similarity score by using deleted elementes", show_default=True)
@click.option("-x", "--xstrings", is_flag=True, help="display similarites of strings")
@click.option("--score", is_flag=True, help="Only display the similarity score for the given APKs. "
        "The flags --deleted and --new still apply")
@click.argument('comp', nargs=2)
def cli(details, diff, compressor, threshold, size, exclude, new, deleted, xstrings, score, comp):
    """
    Compare a Dalvik based file against another file or a whole directory.

    The first argument must be an APK or a single DEX file.
    The second argument might be another APK or DEX or a folder containing
    APK or DEX files.
    In the latter case, the first file is compared against all files in the folder
    recursively.

    If --deleted or --new is not used, a '**' after the numbers indicates
    that these items were not used to calculate the similarity score.
    """
    dx1 = load_analysis(comp[0])
    if dx1 is None:
        raise click.BadParameter("The supplied file '{}' is not an APK or a DEX file!".format(comp[0]))

    FS = FILTERS_DALVIK_SIM
    if exclude:
        FS[elsim.FILTER_SKIPPED_METH].set_regexp(exclude)
    if size:
        FS[elsim.FILTER_SKIPPED_METH].set_size(size)

    if os.path.isdir(comp[1]):
        for root, _, files in os.walk(comp[1], followlinks=True):
            for f in files:
                real_filename = os.path.join(root, f)
                print("filename: %s ..." % real_filename)
                dx2 = load_analysis(real_filename)
                if dx2 is None:
                    click.echo(click.style("The file '{}' is not an APK or DEX. Skipping.".format(real_filename), fg='red'), err=True)
                check_one_file(dx1, dx2, FS, threshold, compressor, details, xstrings, new, deleted, diff, score)
    else:
        dx2 = load_analysis(comp[1])
        if dx2 is None:
            raise click.BadParameter("The supplied file '{}' is not an APK or a DEX file!".format(comp[1]))
        check_one_file(dx1, dx2, FS, threshold, compressor, details, xstrings, new, deleted, diff, score)


if __name__ == "__main__":
    cli()

