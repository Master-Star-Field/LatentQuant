"""
CLI module for embeddings_squeeze package.
"""

import sys
import os

# Add the embeddings_squeeze directory to Python path
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir)

from squeeze import main as squeeze_main

def squeeze():
    """Entry point for squeeze command."""
    squeeze_main()
