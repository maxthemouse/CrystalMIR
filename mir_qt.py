from __future__ import absolute_import
import sys
from PyQt5 import QtGui, uic, QtWidgets
import run_mir_calc
import os

qtCreatorFile = u"main_window.ui"  # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        self.parameter = run_mir_calc.default_param
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        # insert default values
        self.root_dir_val.setText(self.parameter[u"root_dir"])
        self.rc_file_val.setText(self.parameter[u"rc_file"])
        self.output_dir_val.setText(self.parameter[u"output_dir"])
        l, r, t, b = self.parameter[u"crop"]
        self.crop_left.setValue(l)
        self.crop_right.setValue(r)
        self.crop_top.setValue(t)
        self.crop_bottom.setValue(b)
        self.check2pt.setCheckState(self.parameter[u"calc_2"])
        self.check3pt_t.setCheckState(self.parameter[u"calc_3"])
        self.check3pt_g.setCheckState(self.parameter[u"calc_3_gauss"])
        self.checkg_fit.setCheckState(self.parameter[u"calc_gauss"])
        self.check_output_bkg.setCheckState(self.parameter[u"output_bkg"])
        self.check_verbose.setCheckState(self.parameter[u"verbose"])
        self.check_clean_filter.setCheckState(self.parameter[u"clean_filter"])
        min_, max_ = self.parameter[u"clean_filter_limits"]
        self.clean_filter_min.setValue(min_)
        self.clean_filter_max.setValue(max_)
        self.clean_filter_offset.setValue(self.parameter[u"clean_offset"])
        self.check_median.setCheckState(self.parameter[u"median_filter"])
        self.median_size.setValue(self.parameter[u"median_filter_m"])
        self.check_log_abs.setCheckState(self.parameter[u"abs_log"])
        self.check_save_map.setCheckState(self.parameter[u"save_param_map"])
        self.pick_fit_func.addItems([u"pearson", u"gauss"])
        if self.parameter[u"fit_func"] == u"pearson":
            self.pick_fit_func.setCurrentIndex(0)
        else:
            self.pick_fit_func.setCurrentIndex(1)
        dtypes = {u"float": 0, u"uint": 1, u"ubyte": 2}
        self.pick_save_dtype.addItems([u"float", u"uint", u"ubyte"])
        self.pick_fit_func.setCurrentIndex(
            dtypes[self.parameter[u"save_dtype"]])
        self.check_rc_calc.setCheckState(self.parameter[u"calc_rocking_curve"])
        self.rc_energy_kev.setValue(self.parameter[u"rc_energy_kev"])
        self.hkl_index.addItems([u"[2,2,0]", u"[4,4,0]"])
        self.hkl_index.setCurrentIndex(self.parameter[u"hkl_index"])
        # clear and get ready for use
        self.tabWidget.setCurrentIndex(0)
        self.statusBar.clearMessage()
        self.parameter[u"output_text"] = self.statusBar
        self.parameter[u"output_bar"] = ProgressBar(self.lcdNumber)
        self.parameter[u"output_QtGui.qApp"] = QtWidgets.qApp
        self.parameter[u"stop_calc"] = self.checkStop

    def read_values(self):
        self.parameter[u"root_dir"] = unicode(self.root_dir_val.text())
        self.parameter[u"rc_file"] = unicode(self.rc_file_val.text())
        self.parameter[u"output_dir"] = unicode(self.output_dir_val.text())
        l = self.crop_left.value()
        r = self.crop_right.value()
        t = self.crop_top.value()
        b = self.crop_bottom.value()
        self.parameter[u"crop"] = [l, r, t, b]
        self.parameter[u"calc_2"] = self.check2pt.isChecked()
        self.parameter[u"calc_3"] = self.check3pt_t.isChecked()
        self.parameter[u"calc_3_gauss"] = self.check3pt_g.isChecked()
        self.parameter[u"calc_gauss"] = self.checkg_fit.isChecked()
        self.parameter[u"output_bkg"] = self.check_output_bkg.isChecked()
        self.parameter[u"verbose"] = self.check_verbose.isChecked()
        self.parameter[u"clean_filter"] = self.check_clean_filter.isChecked()
        min_ = self.clean_filter_min.value()
        max_ = self.clean_filter_max.value()
        self.parameter[u"clean_filter_limits"] = (min_, max_)
        self.parameter[u"clean_offset"] = self.clean_filter_offset.value()
        self.parameter[u"median_filter"] = self.check_median.isChecked()
        self.parameter[u"median_filter_m"] = self.median_size.value()
        self.parameter[u"abs_log"] = self.check_log_abs.isChecked()
        self.parameter[u"save_param_map"] = self.check_save_map.isChecked()
        fit_func = [u"pearson", u"gauss"]
        self.parameter[u"fit_func"] == fit_func[
            self.pick_fit_func.currentIndex()]
        dtypes = [u"float", u"uint", u"ubyte"]
        self.parameter[u"save_dtype"] = dtypes[
            self.pick_save_dtype.currentIndex()]
        self.parameter[u"calc_rocking_curve"] = self.check_rc_calc.isChecked()
        self.parameter[u"rc_energy_kev"] =self.rc_energy_kev.value()
        self.parameter[u"hkl_index"] = self.hkl_index.currentIndex()

    def start_calc(self):
        # print "Start Calc"
        self.read_values()
        self.parameter[u"paths"] = None  #  reset the path so it will be created again
        # print self.parameter
        run_mir_calc.run_calc(self.parameter)

    def stop_calc(self):
        # signal to stop long calculation
        self.checkStop.setCheckState(True)

    def get_root_dir(self):
        dname = QtWidgets.QFileDialog.getExistingDirectory(self, u'Select folder',
                                                       os.getcwdu())
        self.parameter[u"root_dir"] = dname
        self.root_dir_val.setText(dname)

    def get_rc_file(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, u'Open file',
                                                  os.getcwdu(),
                                                  u"Text files (*.txt)")
        self.parameter[u"rc_file"] = fname
        self.rc_file_val.setText(fname)


class ProgressBar(object):
    u"""Progress bar object to provide common method wrapping for
    using a widget in a gui and pyprind elsewhere."""

    def __init__(self, widget):
        self.bar = widget

    def maximum(self, val):
        pass

    def setValue(self, val):
        self.bar.display(u'%03.2f' % val)

    def reset(self):
        self.bar.display(u'%03.2f' % 0)

    def show(self):
        pass


if __name__ == u"__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
