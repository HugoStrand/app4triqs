# Generated automatically using the command :
# c++2py ../c++/arithmetic.hpp -m arithmetic -a app4triqs -o arithmetic --moduledoc="Example application using Triqs" --cxxflags="-std=c++17" -C pytriqs --include /opt/local/include --include /opt/local/include/openmpi-clang50/
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "arithmetic", doc = "Example application using Triqs", app_name = "app4triqs")

# Imports

# Add here all includes
module.add_include("arithmetic.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""

""")


module.add_function ("double power_of_two (double a)", doc = """""")



module.generate_code()