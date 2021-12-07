import numpy as np


class vtiWriter:
    def __init__(self, p0, p1, originV, dxV):
        self.x1 = p0[0]
        self.y1 = p0[1]
        self.z1 = p0[2]
        self.x2 = p1[0]
        self.y2 = p1[1]
        self.z2 = p1[2]
        self.x0 = originV[0]
        self.y0 = originV[1]
        self.z0 = originV[2]
        self.dx = dxV[0]
        self.dy = dxV[1]
        self.dz = dxV[2]

    def vti_main_header(self, file, as_celldata):
        file.write("<?xml version=\"1.0\"?>\n \
        <VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n \
        <ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"%f %f %f\" Spacing=\"%f %f %f\">\n \
        <Piece Extent=\"%d %d %d %d %d %d\">\n"
                   % (self.x1, self.x2, self.y1, self.y2, self.z1, self.z2, self.x0, self.y0, self.z0, self.dx, self.dy,
                      self.dz, self.x1, self.x2, self.y1, self.y2, self.z1, self.z2))

    def vti_main_footer(self, file):
        file.write("</Piece>\n \
        </ImageData>\n \
       </VTKFile>")

    def vti_celldata_header(self, file, scalar_name):
        file.write("<CellData Scalars=\"%s\">\n" % scalar_name)

    def vti_celldata_footer(self, file):
        file.write("</CellData>\n")

    def vti_pointdata_header(self, file, scalar_name):
        file.write("<PointData Scalars=\"%s\">\n" % scalar_name)

    def vti_pointdata_footer(self, file):
        file.write("</PointData>\n")

    def vti_write_single_data_array(self, file, np_array, array_size, scalar_name):
        file.write("<DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n" % scalar_name)
        for i in range(array_size):
            file.write("%.4E " % (np_array[i]))
        file.write("\n</DataArray>\n")

    def vti_from_txt(self, vti_file, txt_file, as_celldata, scalar_name):
        data = np.loadtxt(txt_file)
        self.vti_from_vector(data, vti_file, as_celldata, scalar_name)

    def vti_from_npy(self, vti_file, npy_file, as_celldata, scalar_name):
        data = np.load(npy_file)
        self.vti_from_vector(data, vti_file, as_celldata, scalar_name)

    def vti_from_vector(self, vti_file, data, as_celldata, scalar_name):
        size = np.shape(data)
        length = size[0]

        vti_file_object = open(vti_file, "w+")

        self.vti_main_header(vti_file_object, as_celldata)

        if as_celldata:
            self.vti_celldata_header(vti_file_object, scalar_name)
        else:
            self.vti_pointdata_header(vti_file_object, scalar_name)

        self.vti_write_single_data_array(vti_file_object, data, length, scalar_name)

        if as_celldata:
            self.vti_celldata_footer(vti_file_object)
        else:
            self.vti_pointdata_footer(vti_file_object)

        self.vti_main_footer(vti_file_object)

        vti_file_object.close()

    def vti_from_multiple_vector(self, vti_file, data_list, as_celldata_list, scalar_name_list):
        assert len(data_list) == len(scalar_name_list)
        assert len(data_list) == len(as_celldata_list)

        vti_file_object = open(vti_file, "w+")

        self.vti_main_header(vti_file_object, as_celldata_list)

        as_celldata = as_celldata_list[0]
        scalar_name = scalar_name_list[0]
        if as_celldata:
            self.vti_celldata_header(vti_file_object, scalar_name)
        else:
            self.vti_pointdata_header(vti_file_object, scalar_name)

        for i in range(len(scalar_name_list)):
            data = data_list[i]
            scalar_name = scalar_name_list[i]
            as_celldata = as_celldata_list[i]

            size = np.shape(data)
            length = size[0]

            self.vti_write_single_data_array(vti_file_object, data, length, scalar_name)

        if as_celldata:
            self.vti_celldata_footer(vti_file_object)
        else:
            self.vti_pointdata_footer(vti_file_object)

        self.vti_main_footer(vti_file_object)

        vti_file_object.close()


def vti_main_header(file, v_extent, v_origin, v_dx):
    x1 = v_extent[0]
    x2 = v_extent[1]
    y1 = v_extent[2]
    y2 = v_extent[3]
    z1 = v_extent[4]
    z2 = v_extent[5]

    x0 = v_origin[0]
    y0 = v_origin[1]
    z0 = v_origin[2]

    dx = v_dx[0]
    dy = v_dx[1]
    dz = v_dx[2]

    file.write("<?xml version=\"1.0\"?>\n \
    <VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n \
    <ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"%f %f %f\" Spacing=\"%f %f %f\">\n \
    <Piece Extent=\"%d %d %d %d %d %d\">\n" % (
        x1, x2, y1, y2, z1, z2, x0, y0, z0, dx, dy, dz, x1, x2, y1, y2, z1, z2))


def vti_main_footer(file):
    file.write("</Piece>\n \
    </ImageData>\n \
   </VTKFile>")


def vti_celldata_header(file, scalar_name):
    file.write("<CellData Scalars=\"%s\">\n" % scalar_name)


def vti_celldata_footer(file):
    file.write("</CellData>\n")


def vti_pointdata_header(file, scalar_name):
    file.write("<PointData Scalars=\"%s\">\n" % scalar_name)


def vti_pointdata_footer(file):
    file.write("</PointData>\n")


def vti_write_single_data_array(file, np_array, array_size, scalar_name):
    file.write("<DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n" % scalar_name)
    for i in range(array_size):
        file.write("%.4E " % (np_array[i]))
    file.write("\n</DataArray>\n")


def vti_from_txt(txt_file, vti_file, v_extent, v_origin, v_dx):
    data = np.loadtxt(txt_file)
    size = np.shape(data)
    length = size[0]

    vti_file_object = open(vti_file, "w+")

    vti_main_header(vti_file_object, v_extent, v_origin, v_dx)
    vti_write_single_data_array(vti_file_object, data, length)
    vti_main_footer(vti_file_object)

    vti_file_object.close()


def vti_from_npy(npy_file, vti_file, v_extent, v_origin, v_dx):
    data = np.load(npy_file)
    size = np.shape(data)
    length = size[0]

    vti_file_object = open(vti_file, "w+")

    vti_main_header(vti_file_object, v_extent, v_origin, v_dx)
    vti_write_single_data_array(vti_file_object, data, length)
    vti_main_footer(vti_file_object)

    vti_file_object.close()


def vti_from_vector(data, vti_file, v_extent, v_origin, v_dx):
    size = np.shape(data)
    length = size[0]

    vti_file_object = open(vti_file, "w+")

    vti_main_header(vti_file_object, v_extent, v_origin, v_dx)
    vti_write_single_data_array(vti_file_object, data, length)
    vti_main_footer(vti_file_object)

    vti_file_object.close()

# HOW TO USE
# For z-slice
# v_extent = [0, Nx, 0, Ny, 0, 0]
# v_origin = [0, 0, z_value]
# v_dx = [dx, dy, 0.0]
#
# vti_from_txt(inpfile, outfile, v_extent, v_origin, v_dx)
# vti_from_npy(inpfile, outfile, v_extent, v_origin, v_dx)
