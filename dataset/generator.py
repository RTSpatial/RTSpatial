#!/usr/bin/env python3
from abc import ABC, abstractmethod # Abstract Base Class
import cgi
import cgitb
import sys # To access stdout
import random as rand # For random number generation
import math # For sin, cos, pi and log functions
from dataclasses import dataclass # To create data classes
import os
import bz2 # For compressing the output to BZ2

# Genreate a random number in the range [min, max)
def uniform(min, max):
    return rand.random() * (max - min) + min

# Generate a random number from a Bernoulli distribution
def bernoulli(p):
    return 1 if rand.random() < p else 0

# Generate a random number from a normal distribution with the given mean and standard deviation
def normal(mu, sigma):
    return mu + sigma * math.sqrt(-2 * math.log(rand.random())) * math.sin(2 * math.pi * rand.random())

# Generate a random integer number in the range [1, n]
def dice(n):
    return math.floor(rand.random() * n) + 1

# A class that accepts geometries and writes them to the appropriate output
class DataSink(ABC):
    @abstractmethod
    def writePoint(self, coordinates):
        pass

    @abstractmethod
    def writeBox(self, coordinates):
        pass
#add one for polygon
    @abstractmethod
    def writePolygon(self, coordinates):
        pass
    @abstractmethod
    def flush(self):
        pass

class CSVSink(DataSink):
    def __init__(self, output):
        self.output = output
    
    def writePoint(self, coordinates):
        self.output.write(",".join([str(elem) for elem in coordinates]))
        self.output.write("\n")

    def writeBox(self, minCoordinates, maxCoordinates):
        self.output.write(",".join([str(elem) for elem in minCoordinates]))
        self.output.write(",")
        self.output.write(",".join([str(elem) for elem in maxCoordinates]))
        self.output.write("\n")
    #add one for polygon
    def writePolygon(self, coordinates):
        for coord in coordinates:
            self.output.write(",".join([str(elem) for elem in coord]))
            self.output.write(";")
        self.output.write(f"{coordinates[0][0]},{coordinates[0][1]}")
        self.output.write("\n")
    def flush(self):
        self.output.flush()

class WKTSink(DataSink):
    def __init__(self, output):
        self.output = output
    
    def writePoint(self, coordinates):
        self.output.write("POINT(")
        self.output.write(" ".join([str(elem) for elem in coordinates]))
        self.output.write(")\n")

    def writeBox(self, minCoordinates, maxCoordinates):
        self.output.write("POLYGON((")
        self.output.write(f"{minCoordinates[0]} {minCoordinates[1]},")
        self.output.write(f"{maxCoordinates[0]} {minCoordinates[1]},")
        self.output.write(f"{maxCoordinates[0]} {maxCoordinates[1]},")
        self.output.write(f"{minCoordinates[0]} {maxCoordinates[1]},")
        self.output.write(f"{minCoordinates[0]} {minCoordinates[1]}")
        self.output.write("))\n")
    def writePolygon(self, coordinates):
        self.output.write("POLYGON((")
        for coord in coordinates:
            self.output.write(" ".join([str(elem) for elem in coord]))
            self.output.write(",")
        self.output.write(f"{coordinates[0][0]} {coordinates[0][1]}")
        self.output.write("))\n")

    def flush(self):
        self.output.flush()

class GeoJSONSink(DataSink):
    def __init__(self, output):
        self.output = output
        self.first_record = True
        self.output.write('{"type": "FeatureCollection", "features": [')
    
    def writePoint(self, coordinates):
        if not self.first_record:
            self.output.write(",")
        self.output.write("\n")
        self.output.write('{"type": "Feature", "geometry": { "type": "Point", "coordinates": [')
        self.output.write(",".join([str(elem) for elem in coordinates]))
        self.output.write("]} }")
        self.first_record = False
        

    def writeBox(self, minCoordinates, maxCoordinates):
        if not self.first_record:
            self.output.write(",")
        self.output.write("\n")
        self.output.write('{"type": "Feature", "geometry": { "type": "Polygon", "coordinates": [[')
        self.output.write(f"[{minCoordinates[0]},{minCoordinates[1]}],")
        self.output.write(f"[{maxCoordinates[0]},{minCoordinates[1]}],")
        self.output.write(f"[{maxCoordinates[0]},{maxCoordinates[1]}],")
        self.output.write(f"[{minCoordinates[0]},{maxCoordinates[1]}],")
        self.output.write(f"[{minCoordinates[0]},{minCoordinates[1]}]")
        self.output.write("]]} }")
        self.first_record = False
    def writePolygon(self, coordinates):
        if not self.first_record:
            self.output.write(",")
        self.output.write("\n")
        self.output.write('{"type": "Feature", "geometry": { "type": "Polygon", "coordinates": [[')
        #array for coordinates
        temp1 = []
        for coord in coordinates:
            #joins each of the array elements in array coordinates into a string separated by a comma
            coords = ",".join([str(elem) for elem in coord])
            out = f"[{coords}]"
            #appends formatted string into the array
            temp1.append(out)
        #formats first array in array coordinates to append at the end
        lastCoord= ",".join([str(elem) for elem in coordinates[0]])
        temp1.append(f"[{lastCoord}]")
        self.output.write(",".join(temp1))
        self.output.write("]]} }")
        self.first_record = False

        #self.output.write(" }")
    def flush(self):
        self.output.write("]}")
        self.output.flush()

# A data sink that takes points and converts them to boxes
class PointToBoxSink(DataSink):
    def __init__(self, sink, maxsize):
        self.sink = sink
        self.maxsize = maxsize
    
    def writePoint(self, coordinates):
        # Generate a box around the coordinates
        minCoordinates = []
        maxCoordinates = []
        for d in range(0, len(coordinates)):
            size = uniform(0, self.maxsize[d])
            minCoordinates.append(coordinates[d] - size)
            maxCoordinates.append(coordinates[d] + size)
        self.writeBox(minCoordinates, maxCoordinates)
    
    def writeBox(self, minCoordinates, maxCoordinates):
        self.sink.writeBox(minCoordinates, maxCoordinates)
    
    def writePolygon(self, coordinates):
        sys.stdout.write("writing a polygon")
        #google what join does
    def flush(self):
        self.sink.flush()
class PointToPolygonSink(DataSink):
    def __init__(self, sink, maxseg, polysize):
        self.sink = sink
        self.maxseg = maxseg
        self.polysize = polysize
    def transform(self, center, angle):
        x = center[0] + self.polysize * math.cos(angle)
        y = center[1] + self.polysize * math.sin(angle)
        return [x, y]
    def flush(self):
        self.sink.flush()

    def writePoint(self, coordinates):
        center = coordinates
        minSegs = 3
        if(self.maxseg <= 3):
            numSegments = minSegs
        else:
            numSegments = dice(self.maxseg - minSegs) + minSegs
        angles = []
        for increment in range(0, numSegments):
            angles.append(uniform(0, math.pi * 2))
        angles.sort()
        points = []
        for angle in angles:
            points.append(self.transform(center, angle))
        self.writePolygon(points)
    
    def writeBox(self, coordinates):
        print("got to write box")
    
    def writePolygon(self, coordinates):
        self.sink.writePolygon(coordinates)

class BZ2OutputStream:
    def __init__(self, output):
        self.output = output
        self.compressor = bz2.BZ2Compressor()
    
    def write(self, data):
        compressedData = self.compressor.compress(bytes(data, "utf-8"))
        self.output.write(compressedData)
    
    def flush(self):
        data = self.compressor.flush() # Get the last bits of data remaining in the compressor
        self.output.write(data)
        self.output.flush()

# A data sink that converts all shapes using an affine transformation before writing them
class AffineTransformSink(DataSink):
    def __init__(self, sink, dim, affineMatrix):
        self.sink = sink
        assert len(affineMatrix) == dim * (dim + 1)
        squarematrix = []
        for d in range(0, dim):
            squarematrix.append(affineMatrix[d * (dim+1) : (d+1) * (dim+1)])
        squarematrix.append([0]* dim + [1])
        self.affineMatrix = squarematrix
    
    # Transform a point using an affine transformation matrix
    def affineTransformPoint(self, coordinates):
        # Transform the array

        # The next line uses numpy for matrix multiplication. But we use our own code to reduce the dependency
        # Append [1] to the input cordinates to match the affine transformation
        # Remove the last element from the result
        # transformed = np.matmul(self.affineMatrix, coordinates + [1])[:-1]

        # Matrix multiplication using a regular code
        dim = len(coordinates)
        transformed = [0] * dim
        for i in range(0, dim):
            transformed[i] = self.affineMatrix[i][dim]
            for d in range(0, dim):
                transformed[i] += coordinates[d] * self.affineMatrix[i][d]

        return transformed
    
    def writePoint(self, coordinates):
        self.sink.writePoint(self.affineTransformPoint(coordinates))

    def writeBox(self, minCoordinates, maxCoordinates):
        self.sink.writeBox(self.affineTransformPoint(minCoordinates), self.affineTransformPoint(maxCoordinates))
    
    def writePolygon(self, coordinates):
        for coord in coordinates:
            self.sink.writeBox(self.affineTransformPoint(coord[0]), self.affineTransformPoint(coord[1]))

    def flush(self):
        self.sink.flush()

# An abstract generator
class Generator(ABC):
    
    def __init__(self, card, dim):
        self.card = card
        self.dim = dim

    # Set the sink to which generated records will be written
    def setSink(self, datasink):
        self.datasink = datasink

    # Check if the given point is valid, i.e., all coordinates in the range [0, 1]
    def isValidPoint(self, point):
        for x in point:
            if not (0 <= x <= 1):
                return False
        return True

    # Generate all points and write them to the data sink
    @abstractmethod
    def generate(self):
        pass

class PointGenerator(Generator):
    def __init__(self, card, dim):
        super(PointGenerator, self).__init__(card, dim)
    
    @abstractmethod
    def generatePoint(self, i, prevpoint):
        pass

    def generate(self):
        i = 0
        prevpoint = None
        while i < self.card:
            newpoint = self.generatePoint(i, prevpoint)
            if self.isValidPoint(newpoint):
                self.datasink.writePoint(newpoint)
                prevpoint = newpoint
                i += 1
        self.datasink.flush()

# Generate uniformly distributed points
class UniformGenerator(PointGenerator):

    def __init__(self, card, dim):
        super(UniformGenerator, self).__init__(card, dim)

    def generatePoint(self, i, prev_point):
        return [rand.random() for d in range(self.dim)]

# Generate points from a diagonal distribution
class DiagonalGenerator(PointGenerator):

    def __init__(self, card, dim, percentage, buffer):
        super(DiagonalGenerator, self).__init__(card, dim)
        self.percentage = percentage
        self.buffer = buffer

    def generatePoint(self, i, prev_point):
        if bernoulli(self.percentage) == 1:
            return [rand.random()] * self.dim
        else:
            c = rand.random()
            d = normal(0, self.buffer / 5)
            return [(c + (1 - 2 * (x % 2)) * d / math.sqrt(2)) for x in range(self.dim)]

class GaussianGenerator(PointGenerator):
    def __init__(self, card, dim):
        super(GaussianGenerator, self).__init__(card, dim)

    def generatePoint(self, i, prev_point):
        return [normal(0.5, 0.1) for d in range(self.dim)]

class SierpinskiGenerator(PointGenerator):
    def __init__(self, card, dim):
        super(SierpinskiGenerator, self).__init__(card, dim)

    def generatePoint(self, i, prev_point):
        if i == 0:
            return [0.0, 0.0]
        elif i == 1:
            return [1.0, 0.0]
        elif i == 2:
            return [0.5, math.sqrt(3) / 2]
        else:
            d = dice(5)

            if d == 1 or d == 2:
                return self.get_middle_point(prev_point, [0.0, 0.0])
            elif d == 3 or d == 4:
                return self.get_middle_point(prev_point, [1.0, 0.0])
            else:
                return self.get_middle_point(prev_point, [0.5, math.sqrt(3) / 2])

    def get_middle_point(self, point1, point2):
        middle_point_coords = []
        for i in range(len(point1)):
            middle_point_coords.append((point1[i] + point2[i]) / 2)
        return middle_point_coords

class BitGenerator(PointGenerator):
    def __init__(self, card, dim, prob, digits):
        super(BitGenerator, self).__init__(card, dim)
        self.prob = prob
        self.digits = digits

    def generatePoint(self, i, prev_point):
        return [self.bit() for d in range(self.dim)]

    def bit(self):
        num = 0.0
        for i in range(1, self.digits + 1):
            c = bernoulli(self.prob)
            num = num + c / (math.pow(2, i))
        return num

# A two-dimensional box with depth field. Used with the parcel generator
@dataclass
class BoxWithDepth:
    depth: int
    x: float
    y: float
    w: float
    h: float

class ParcelGenerator(Generator):
    def __init__(self, card, dim, split_range, dither):
        super(ParcelGenerator, self).__init__(card, dim)
        self.split_range = split_range
        self.dither = dither

    def generate(self):
        # Using dataclass to create BoxWithDepth, which stores depth of each box in the tree
        # Depth is used to determine at which level to stop splitting and start printing    
        box = BoxWithDepth(0, 0.0, 0.0, 1.0, 1.0)
        boxes = [] # Empty stack for depth-first generation of boxes
        boxes.append(box)
        
        max_height = math.ceil(math.log(self.card, 2))
        
        # We will print some boxes at last level and the remaining at the second to last level 
        # Number of boxes to split on the second to last level
        numToSplit = self.card - pow(2, max(max_height - 1, 0))
        numSplit = 0
        boxes_generated = 0

        while boxes_generated < self.card:
            b = boxes.pop()
            
            if b.depth >= max_height - 1:
                if numSplit < numToSplit: # Split at second to last level and print the new boxes
                    b1, b2 = self.split(b, boxes)
                    numSplit += 1
                    self.dither_and_print(b1)
                    self.dither_and_print(b2)
                    boxes_generated += 2
                else: # Print remaining boxes from the second to last level 
                    self.dither_and_print(b)
                    boxes_generated += 1
                    if boxes_generated == 10: # Early flush to ensure immediate download of data
                        sys.stdout.buffer.flush()
  
            else:
                b1, b2 = self.split(b, boxes)
                boxes.append(b2)
                boxes.append(b1)
        self.datasink.flush()
            
    def split(self, b, boxes):
        if b.w > b.h:
            # Split vertically if width is bigger than height
            # Tried numpy random number generator, found to be twice as slow as the Python default generator
            split_size = b.w * uniform(self.split_range, 1 - self.split_range)
            b1 = BoxWithDepth(b.depth+1,b.x, b.y, split_size, b.h)
            b2 = BoxWithDepth(b.depth + 1, b.x + split_size, b.y, b.w - split_size, b.h)
        else:
            # Split horizontally if width is less than height
            split_size = b.h * uniform(self.split_range, 1 - self.split_range)
            b1 = BoxWithDepth(b.depth+1, b.x, b.y, b.w, split_size)
            b2 = BoxWithDepth(b.depth+1, b.x, b.y + split_size, b.w, b.h - split_size) 
        return b1, b2
    
    def dither_and_print(self, b):
        ditherx = b.w * uniform(0.0, self.dither)
        b.x += ditherx / 2
        b.w -= ditherx
        dithery = b.h * uniform(0.0, self.dither)
        b.y += dithery / 2
        b.h -= dithery

        self.datasink.writeBox([b.x, b.y], [b.x + b.w, b.y + b.h])
        
    def generate_point(self, i, prev_point):
        raise Exception("Cannot generate points with the ParcelGenerator")

def printUsage():
    sys.stderr.write(f"Usage: {sys.argv[0]} <key1=value1> ... \n")
    sys.stderr.write("The keys and values are listed below")
    sys.stderr.write("distribution: {uniform, diagonal, gaussian, parcel, bit, sierpinski}\n")
    sys.stderr.write("cardinality: Number of geometries to generate\n")
    sys.stderr.write("dimensions: Number of dimensions in generated geometries\n")
    sys.stderr.write("geometry: {point, box}\n")
    sys.stderr.write(" ** if geometry type is 'box' and the distribution is NOT 'parcel', you have to specify the maxsize property\n")
    sys.stderr.write("maxsize: maximum size along each dimension (before transformation), e.g., 0.2,0.2 (no spaces)\n")
    sys.stderr.write("percentage: (for diagonal distribution) the percentage of records that are perfectly on the diagonal\n")
    sys.stderr.write("buffer: (for diagonal distribution) the buffer around the diagonal that additional points can be in\n")
    sys.stderr.write("srange: (for parcel distribution) the split range [0.0, 1.0]\n")
    sys.stderr.write("dither: (for parcel distribution) the amound of noise added to each record as a perctange of its initial size [0.0, 1.0]\n")
    sys.stderr.write("affinematrix: (optional) values of the affine matrix separated by comma. Number of expected values is d*(d+1) where d is the number of dimensions\n")
    sys.stderr.write("compress: (optional) { bz2 }\n")
    sys.stderr.write("format: output format { csv, wkt, geojson }\n")
    sys.stderr.write("[affine matrix] (Optional) Affine matrix parameters to apply to all generated geometries\n")

class CommandLineArguments:
    def __init__(self, argv):
        self.argv = argv
    
    def getvalue(self, name):
        for arg in self.argv:
            parts = arg.split("=")
            if parts[0] == name:
                return parts[1]
        return None

def main():
    if 'REQUEST_METHOD' in os.environ :
        # This is running from a web page
        httpResult = True
        # Show debugging information on error
        #cgitb.enable()
        form = cgi.FieldStorage()
    else:
        # Running from command line
        httpResult = False
        # Extract parameters from command line
        if len(sys.argv) < 2:
            printUsage()
            sys.exit(1)
        form = CommandLineArguments(sys.argv)

    distribution = form.getvalue("distribution")
    cardinality = int(form.getvalue("cardinality"))
    dimensions = int(form.getvalue("dimensions"))
    geometryType = form.getvalue("geometry")
    generator = None
    if (distribution == "uniform"):
        generator = UniformGenerator(cardinality, dimensions)
    elif (distribution == "diagonal"):
        percentage = float(form.getvalue("percentage"))
        buffer = float(form.getvalue("buffer"))
        generator = DiagonalGenerator(cardinality, dimensions, percentage, buffer)
    elif (distribution == "gaussian"):
        generator = GaussianGenerator(cardinality, dimensions)
    elif (distribution == "parcel"):
        split_range = float(form.getvalue("srange"))
        dither = float(form.getvalue("dither"))
        generator = ParcelGenerator(cardinality, dimensions, split_range, dither)
    elif (distribution == "bit"):
        probability = float(form.getvalue("probability"))
        digits = int(form.getvalue("digits"))
        generator = BitGenerator(cardinality, dimensions, probability, digits)
    elif (distribution == "sierpinski"):
        generator = SierpinskiGenerator(cardinality, dimensions)
    
    compress = form.getvalue("compress")
    output_format = (form.getvalue("format") or "csv").lower()

    if (compress is None):
        output = sys.stdout
    elif (compress == "bz2"):
        if httpResult:
            sys.stdout.buffer.write(bytes("Status: 200 OK\r\n", 'utf-8'))
            sys.stdout.buffer.write(bytes("Content-type: application/x-bzip2\r\n", 'utf-8'))
            #sys.stdout.buffer.write(bytes("Transfer-Encoding: chunked\r\n", 'utf-8'))
            filename = f"{distribution}.{output_format}.bz2"
            sys.stdout.buffer.write(bytes(f"Content-Disposition: attachment; filename=\"{filename}\"\r\n\r\n", 'utf-8'))
        output = BZ2OutputStream(sys.stdout.buffer)
    else:
        raise Exception(f"Unsupported compression '{compress}''")
    
    if (output_format == "wkt"):
        if httpResult and compress is None:
            print("Status: 200 OK")
            print("Content-Type: text/csv")
            print("")
        datasink = WKTSink(output)
    elif (output_format == "csv"):
        if httpResult and compress is None:
            print("Status: 200 OK")
            print("Content-Type: text/csv")
            print("")
        datasink = CSVSink(output)
    elif (output_format == "geojson"):
        if httpResult and compress is None:
            print("Status: 200 OK")
            print("Content-Type: application/geo+json")
            print("")
        datasink = GeoJSONSink(output)
    else:
        raise Exception(f"Unsupported format '{output_format}'")

    if form.getvalue("affinematrix") is not None:
        affineMatrix = [float(x) for x in form.getvalue("affinematrix").split(",") ]
    else:
        affineMatrix = None
    
    if form.getvalue("seed") is not None:
        rand.seed(int(form.getvalue("seed")))
    
    # Connect a point to box converter if the distribution only generated point but boxes are requested
    if (geometryType == 'box' and distribution != 'parcel'):
        maxsize = [float(x) for x in form.getvalue("maxsize").split(",")]
        datasink = PointToBoxSink(datasink, maxsize)
    if (geometryType == 'polygon' and distribution != 'parcel'):
        polysize = float(form.getvalue("polysize"))
        maxseg = float(form.getvalue("maxseg"))
        datasink = PointToPolygonSink(datasink, maxseg, polysize)
    

    # If the number of parmaeters for the affineMatrix is correct, apply the affine transformation
    if (affineMatrix is not None and len(affineMatrix) == dimensions * (dimensions + 1)):
        affineMatrix = [float(x) for x in form.getvalue("affinematrix").split(",")]
        datasink = AffineTransformSink(datasink, dimensions, affineMatrix)
    
    # Set the data sink (receiver) and run the generator
    generator.setSink(datasink)
    generator.generate()

if __name__ == "__main__":
    main()
 