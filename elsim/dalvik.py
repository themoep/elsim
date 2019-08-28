"""
This module encapsultes dalvik code for the use with elsim
"""
# This file is part of Elsim
#
# Copyright (C) 2012, Anthony Desnos <desnos at t0t0.fr>
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

import re
from operator import itemgetter
import mmh3

from androguard.core.bytecodes import dvm

from elsim import debug
import elsim
from elsim.filters import filter_sort_meth_basic, FilterNone


# FIXME: what was this?!
DEFAULT_SIGNATURE = 'L0_4'


class FilterSkip:
    def __init__(self, size=1, regexp=None):
        # Minimal size of one should always be the case. We can not compare to empty strings.
        self.size = size
        self.regexp = regexp

    def skip(self, m):
        if m.get_length() < self.size:
            return True

        if self.regexp and re.match(self.regexp, m.m.get_class_name()):
            return True

        return False

    def set_regexp(self, e):
        self.regexp = e

    def set_size(self, e):
        """
        :param int e: the minimal size:
        """
        if e < 0:
            raise ValueError("size must be positive integer")
        self.size = e


FILTERS_DALVIK_SIM = {
    # input element, iterator
    elsim.FILTER_ELEMENT_METH: lambda element, iterator: Method(iterator.vmx, element),
    elsim.FILTER_CHECKSUM_METH: lambda element, sim: CheckSumMeth(element, sim),
    elsim.FILTER_SIM_METH: lambda sim, e1, e2: sim.ncd(e1.checksum.get_signature(), e2.checksum.get_signature()),
    elsim.FILTER_SORT_METH: filter_sort_meth_basic,
    elsim.FILTER_SKIPPED_METH: FilterSkip(),
}

FILTERS_DALVIK_SIM_STRING = {
        elsim.FILTER_ELEMENT_METH: lambda element, iterator: StringVM(element),
        elsim.FILTER_CHECKSUM_METH: lambda element, sim: CheckSumString(element, sim),
        elsim.FILTER_SIM_METH: lambda sim, e1, e2: sim.ncd(e1.checksum.get_buff(), e2.checksum.get_buff()),
    elsim.FILTER_SORT_METH: filter_sort_meth_basic,
    elsim.FILTER_SKIPPED_METH: FilterNone,
}

FILTERS_DALVIK_BB = {
    # input element, iterator
    # FIXME: this might be iterator instead?
    elsim.FILTER_ELEMENT_METH: lambda element, iterator: BasicBlock(element),
    elsim.FILTER_CHECKSUM_METH: lambda element, sim: CheckSumBB(element, sim),
    elsim.FILTER_SIM_METH: lambda sim, e1, e2: sim.ncd(e1.checksum.get_buff(), e2.checksum.get_buff()),
    elsim.FILTER_SORT_METH: filter_sort_meth_basic,
    elsim.FILTER_SKIPPED_METH: FilterNone,
}



class CheckSumMeth:
    def __init__(self, m1, sim):
        """
        :param Method m1:
        :param elsim.similarity.Similarity sim:
        """
        self.m1 = m1
        self.sim = sim

        self.buff = ""
        self.signature = None
        self.signature_entropy = None

        # This essentially creates a long string with
        # all the instructions as names plus their operands in
        # a human readable form
        for i in m1.m.get_instructions():
            self.buff += dvm.clean_name_instruction(i)
            self.buff += dvm.static_operand_instruction(i)

        self.buff = self.buff.encode('UTF-8')
        self.entropy = sim.entropy(self.buff)

        # FIXME: as long as we dont have Signature back online, we use the bytecode directly
        if self.m1.m.get_code():
            self.signature = self.m1.m.get_code().get_bc().get_insn()
            self.signature_entropy = self.sim.entropy(self.signature)
        else:
            self.signature = b''
            self.signature_entropy = 0.0

    def get_signature(self):
        """
        The Signature proposed here is an Android Variant of
        Cesare and Xiang (2010): Classification of Malware Using Structured Control Flow
        
        You can also read about this in http://phrack.org/issues/68/15.html
        """
        if self.signature == None:
            # FIXME
            self.signature = self.m1.vmx.get_method_signature(self.m1.m, predef_sign=DEFAULT_SIGNATURE).get_string()
            self.signature_entropy = self.sim.entropy(self.signature)

        return self.signature

    def get_signature_entropy(self):
        if self.signature == None:
            # FIXME
            self.signature = self.m1.vmx.get_method_signature(self.m1.m, predef_sign=DEFAULT_SIGNATURE).get_string()
            self.signature_entropy = self.sim.entropy(self.signature)

        return self.signature_entropy

    def get_entropy(self):
        return self.entropy

    def get_buff(self):
        return self.buff


class CheckSumBB:
    def __init__(self, basic_block, sim):
        self.basic_block = basic_block
        self.buff = ""
        for i in self.basic_block.bb.get_instructions():
            self.buff += dvm.clean_name_instruction(i)
            self.buff += dvm.static_operand_instruction(i)

        self.buff = self.buff.encode('UTF-8')
        self.hash = mmh3.hash128(self.buff)

    def get_buff(self):
        return self.buff

    def get_hash(self):
        return self.hash


DIFF_INS_TAG = {
    "ORIG": 0,
    "ADD": 1,
    "REMOVE": 2
}


class DiffBB:
    def __init__(self, bb1, bb2, info):
        self.bb1 = bb1
        self.bb2 = bb2
        self.info = info

        self.start = self.bb1.start
        self.end = self.bb1.end
        self.name = self.bb1.name

        self.di = None
        self.ins = []

    def diff_ins(self, di):
        self.di = di

        off_add = {}
        off_rm = {}
        for i in self.di.add_ins:
            off_add[i[0]] = i

        for i in self.di.remove_ins:
            off_rm[i[0]] = i

        nb = 0
        for i in self.bb1.ins:
            ok = False
            if nb in off_add:
                debug("%d ADD %s %s" % (
                    nb, off_add[nb][2].get_name(), off_add[nb][2].get_output()))
                self.ins.append(off_add[nb][2])
                setattr(off_add[nb][2], "diff_tag", DIFF_INS_TAG["ADD"])
                del off_add[nb]

            if nb in off_rm:
                debug("%d RM %s %s" %
                      (nb, off_rm[nb][2].get_name(), off_rm[nb][2].get_output()))
                self.ins.append(off_rm[nb][2])
                setattr(off_rm[nb][2], "diff_tag", DIFF_INS_TAG["REMOVE"])
                del off_rm[nb]
                ok = True

            if ok == False:
                self.ins.append(i)
                debug("%d %s %s" % (nb, i.get_name(), i.get_output()))
                setattr(i, "diff_tag", DIFF_INS_TAG["ORIG"])

            nb += 1

        nbmax = nb
        if off_add != {}:
            nbmax = sorted(off_add)[-1]
        if off_rm != {}:
            nbmax = max(nbmax, sorted(off_rm)[-1])

        while nb <= nbmax:
            if nb in off_add:
                debug("%d ADD %s %s" % (
                    nb, off_add[nb][2].get_name(), off_add[nb][2].get_output()))
                self.ins.append(off_add[nb][2])
                setattr(off_add[nb][2], "diff_tag", DIFF_INS_TAG["ADD"])
                del off_add[nb]

            if nb in off_rm:
                debug("%d RM %s %s" %
                      (nb, off_rm[nb][2].get_name(), off_rm[nb][2].get_output()))
                self.ins.append(off_rm[nb][2])
                setattr(off_rm[nb][2], "diff_tag", DIFF_INS_TAG["REMOVE"])
                del off_rm[nb]

            nb += 1

    def set_childs(self, abb):
        self.childs = self.bb1.childs

        for i in self.ins:
            if i == self.bb2.ins[-1]:
                childs = []
                for c in self.bb2.childs:
                    if c[2].name in abb:
                        debug("SET %s %s" % (c[2], abb[c[2].name]))
                        childs.append((c[0], c[1], abb[c[2].name]))
                    else:
                        debug("SET ORIG %s" % str(c))
                        childs.append(c)

                i.childs = childs

    def show(self):
        print("\tADD INSTRUCTIONS :")
        for i in self.di.add_ins:
            print("\t\t", i[0], i[1], i[2].get_name(), i[2].get_output())

        print("\tREMOVE INSTRUCTIONS :")
        for i in self.di.remove_ins:
            print("\t\t", i[0], i[1], i[2].get_name(), i[2].get_output())


class NewBB:
    def __init__(self, bb):
        self.bb = bb

        self.start = self.bb.start
        self.end = self.bb.end
        self.name = self.bb.name
        self.ins = self.bb.ins

    def set_childs(self, abb):
        childs = []
        for c in self.bb.childs:
            if c[2].name in abb:
                debug("SET %s %s " % (c[2], abb[c[2].name]))
                childs.append((c[0], c[1], abb[c[2].name]))
            else:
                debug("SET ORIG %s" % str(c))
                childs.append(c)

        self.childs = childs


class DiffINS:
    def __init__(self, add_ins, remove_ins):
        self.add_ins = add_ins
        self.remove_ins = remove_ins


DIFF_BB_TAG = {
    "ORIG": 0,
    "DIFF": 1,
    "NEW": 2
}


class Method:
    """
    This object is used to calculate the similarity to another EncodedMethod
    """
    # FIXME: this thing contains so much stuff from the diffing thing...
    def __init__(self, vmx, m):
        """

        :param androguard.core.analysis.analysis.Analysis vmx:
        :param androguard.core.bytecodes.dvm.EncodedMethod m:
        """
        self.m = m
        self.vmx = vmx
        self.mx = vmx.get_method(m)

        self.sort_h = []

        self.__hash = None

    def __str__(self):
        return "%s %s %s %d" % (self.m.get_class_name(), self.m.get_name(), self.m.get_descriptor(), self.m.get_length())

    def get_length(self):
        """Returns the length of the code of the method"""
        return self.m.get_length()

    def set_checksum(self, fm):
        self.__hash = mmh3.hash128(fm.get_buff())
        self.checksum = fm

    @property
    def hash(self):
        return self.__hash

    def diff(self, func_sim_bb, func_diff_ins):
        if self.sort_h == []:
            self.dbb = {}
            self.nbb = {}
            return

        bb1 = self.bb

        # Dict for diff basic blocks
        # vm1 basic block : vm2 basic blocks -> value (0.0 to 1.0)
        diff_bb = {}

        # List to get directly all diff basic blocks
        direct_diff_bb = []

        # Dict for new basic blocks
        new_bb = {}

        # Reverse Dict with matches diff basic blocks
        associated_bb = {}

        for b1 in bb1:
            diff_bb[bb1[b1]] = {}

            debug("%s 0x%x" % (b1, bb1[b1].basic_block.end))
            for i in self.sort_h:
                bb2 = i[0].bb
                b_z = diff_bb[bb1[b1]]

                bb2hash = i[0].bb_sha256

                # If b1 is in bb2:
                # we can have one or more identical basic blocks to b1, we must add them
                if bb1[b1].get_hash() in bb2hash:
                    for equal_bb in bb2hash[bb1[b1].get_hash()]:
                        b_z[equal_bb.basic_block.name] = 0.0

                # If b1 is not in bb2:
                    # we must check similarities between all bb2
                else:
                    for b2 in bb2:
                        b_z[b2] = func_sim_bb(bb1[b1], bb2[b2], self.sim)

                sorted_bb = sorted(b_z.items(), key=itemgetter(1))

                debug("\t\t%s" % sorted_bb[:2])

                for new_diff in sorted_bb:
                    associated_bb[new_diff[0]] = bb1[b1].basic_block

                    if new_diff[1] == 0.0:
                        direct_diff_bb.append(new_diff[0])

                if sorted_bb[0][1] != 0.0:
                    diff_bb[bb1[b1]] = (bb2[sorted_bb[0][0]], sorted_bb[0][1])
                    direct_diff_bb.append(sorted_bb[0][0])
                else:
                    del diff_bb[bb1[b1]]

        for i in self.sort_h:
            bb2 = i[0].bb
            for b2 in bb2:
                if b2 not in direct_diff_bb:
                    new_bb[b2] = bb2[b2]

        dbb = {}
        nbb = {}
        # Add all different basic blocks
        for d in diff_bb:
            dbb[d.basic_block.name] = DiffBB(
                d.basic_block, diff_bb[d][0].basic_block, diff_bb[d])

        # Add all new basic blocks
        for n in new_bb:
            nbb[new_bb[n].basic_block] = NewBB(new_bb[n].basic_block)
            if n in associated_bb:
                del associated_bb[n]

        self.dbb = dbb
        self.nbb = nbb

        # Found diff instructions
        for d in dbb:
            func_diff_ins(dbb[d], self.sim)

        # Set new childs for diff basic blocks
            # The instructions will be tag with a new flag "childs"
        for d in dbb:
            dbb[d].set_childs(associated_bb)

        # Set new childs for new basic blocks
        for d in nbb:
            nbb[d].set_childs(associated_bb)

        # Create and tag all (orig/diff/new) basic blocks
        self.create_bbs()

    def create_bbs(self):
        dbb = self.dbb
        nbb = self.nbb

        # For same block:
        # tag = 0
        # For diff block:
        # tag = 1
        # For new block:
        # tag = 2
        l = []
        for bb in self.mx.basic_blocks.get():
            if bb.name not in dbb:
                # add the original basic block
                bb.bb_tag = DIFF_BB_TAG["ORIG"]
                l.append(bb)
            else:
                # add the diff basic block
                dbb[bb.name].bb_tag = DIFF_BB_TAG["DIFF"]
                l.append(dbb[bb.name])

        for i in nbb:
            # add the new basic block
            nbb[i].bb_tag = DIFF_BB_TAG["NEW"]
            l.append(nbb[i])

        # Sorted basic blocks by addr (orig, new, diff)
        l = sorted(l, key=lambda x: x.start)
        self.bbs = l

    def show(self, details=False, exclude=[]):
        print(self.m.get_class_name(), self.m.get_name(),
              self.m.get_descriptor(), end=' ')
        print("with", end=' ')

        for i in self.sort_h:
            print(i[0].m.get_class_name(), i[0].m.get_name(),
                  i[0].m.get_descriptor(), i[1])

        print("\tDIFF BASIC BLOCKS :")
        for d in self.dbb:
            print("\t\t", self.dbb[d].bb1.name, " --->",
                  self.dbb[d].bb2.name, ":", self.dbb[d].info[1])
            if details:
                self.dbb[d].show()

        print("\tNEW BASIC BLOCKS :")
        for b in self.nbb:
            print("\t\t", self.nbb[b].name)


class BasicBlock:
    def __init__(self, bb):
        self.bb = bb
        self.__hash = None

    def set_checksum(self, fm):
        self.__hash = mmh3.hash128(fm.get_buff())
        self.checksum = fm

    @property
    def hash(self):
        return self.__hash

    def __str__(self):
        return self.bb.name

    def show(self):
        print(self.bb.name)


class StringVM:
    def __init__(self, el):
        self.el = el
        self.__hash = None

    def set_checksum(self, fm):
        # FIXME: we can use the MUTF8 strings here
        self.__hash = mmh3.hash128(fm.get_buff())
        self.checksum = fm

    def get_length(self):
        return len(self.el)

    @property
    def hash(self):
        return self.__hash

    def __str__(self):
        return repr(self.el)


class CheckSumString:
    def __init__(self, m1, sim):
        self.m1 = m1
        self.sim = sim

        self.buff = self.m1.el

    def get_buff(self):
        # FIXME: the checksumstring thing should have a better structure!
        return self.buff.encode('UTF-8')


class ProxyDalvik:
    """
    A simple proxy which uses the methods for comparison
    """
    def __init__(self, vmx):
        """
        :param androgaurd.core.analysis.analysis.Analysis vmx:
        """
        self.vmx = vmx

    def __iter__(self):
        """
        yield many EncodedMethod
        """
        for x in self.vmx.get_methods():
            if not x.is_external():
                yield x.get_method()


class ProxyDalvikMethod:
    """A Proxy for BasicBlocks"""
    def __init__(self, el):
        self.el = el

    def __iter__(self):
        yield from self.el.mx.basic_blocks.get()


class ProxyDalvikString:
    def __init__(self, vmx):
        self.vmx = vmx

    def __iter__(self):
        for i in self.vmx.get_strings():
            yield i.get_value()


def LCS(X, Y):
    """Longest Common Subsequence"""
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n+1) for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C


def getDiff(C, X, Y, i, j, a, r):
    if i > 0 and j > 0 and X[i-1] == Y[j-1]:
        getDiff(C, X, Y, i-1, j-1, a, r)
        debug(" " + "%02X" % ord(X[i-1]))
    else:
        if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
            getDiff(C, X, Y, i, j-1, a, r)
            a.append((j-1, Y[j-1]))
            debug(" + " + "%02X" % ord(Y[j-1]))
        elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
            getDiff(C, X, Y, i-1, j, a, r)
            r.append((i-1, X[i-1]))
            debug(" - " + "%02X" % ord(X[i-1]))


def toString(bb, hS, rS):
    map_x = {}
    S = ""

    idx = 0
    nb = 0
    for i in bb.get_instructions():
        ident = dvm.clean_name_instruction(i)
        ident += dvm.static_operand_instruction(i)

        if ident not in hS:
            hS[ident] = len(hS)
            rS[chr(hS[ident])] = ident

        S += chr(hS[ident])
        map_x[nb] = idx
        idx += i.get_length()
        nb += 1

    return S, map_x


class DiffInstruction:
    def __init__(self, bb, instruction):
        self.bb = bb

        self.pos_instruction = instruction[0]
        self.offset = instruction[1]
        self.ins = instruction[2]

    def show(self):
        print(hex(self.bb.bb.start + self.offset), self.pos_instruction,
              self.ins.get_name(), self.ins.show_buff(self.bb.bb.start + self.offset))


class DiffBasicBlock:
    def __init__(self, x, y, added, deleted):
        self.basic_block_x = x
        self.basic_block_y = y
        self.added = sorted(added, key=lambda x: x[1])
        self.deleted = sorted(deleted, key=lambda x: x[1])

    def get_added_elements(self):
        for i in self.added:
            yield DiffInstruction(self.basic_block_x, i)

    def get_deleted_elements(self):
        for i in self.deleted:
            yield DiffInstruction(self.basic_block_y, i)


def filter_diff_bb(x, y):
    final_add = []
    final_rm = []

    hS = {}
    rS = {}

    X, map_x = toString(x.bb, hS, rS)
    Y, map_y = toString(y.bb, hS, rS)

    debug("%s %d" % (repr(X), len(X)))
    debug("%s %d" % (repr(Y), len(Y)))

    m = len(X)
    n = len(Y)

    C = LCS(X, Y)
    a = []
    r = []

    getDiff(C, X, Y, m, n, a, r)
    debug(a)
    debug(r)

    debug("DEBUG ADD")
    for i in a:
        instructions = [j for j in y.bb.get_instructions()]
        debug(" \t %s %s %s" % (
            i[0], instructions[i[0]].get_name(), instructions[i[0]].get_output()))
        final_add.append((i[0], map_y[i[0]], instructions[i[0]]))

    debug("DEBUG REMOVE")
    for i in r:
        instructions = [j for j in x.bb.get_instructions()]
        debug(" \t %s %s %s" % (
            i[0], instructions[i[0]].get_name(), instructions[i[0]].get_output()))
        final_rm.append((i[0], map_x[i[0]], instructions[i[0]]))

    return DiffBasicBlock(y, x, final_add, final_rm)


FILTERS_DALVIK_DIFF_BB = {
    elsim.DIFF: filter_diff_bb,
}


class ProxyDalvikBasicBlock:
    def __init__(self, esim):
        self.esim = esim

    def __iter__(self):
        x = elsim.split_elements(self.esim, self.esim.get_similar_elements())
        for i in x:
            yield i, x[i]


class DiffDalvikMethod:
    def __init__(self, m1, m2, els, eld):
        self.m1 = m1
        self.m2 = m2
        self.els = els
        self.eld = eld

    def get_info_method(self, m):
        return m.m.get_class_name(), m.m.get_name(), m.m.get_descriptor()

    def show(self):
        print("[", self.get_info_method(self.m1), "]",
              "<->", "[", self.get_info_method(self.m2), "]")

        self.eld.show()

        self.els.show()
        self._show_elements("NEW", self.els.get_new_elements())

    def _show_elements(self, info, elements):
        for i in elements:
            print(i.bb, hex(i.bb.get_start()), hex(i.bb.get_end()))
            idx = i.bb.get_start()
            for j in i.bb.get_instructions():
                print("\t" + info, hex(idx), end=' ')
                j.show(idx)
                print()
                idx += j.get_length()

        print("\n")

