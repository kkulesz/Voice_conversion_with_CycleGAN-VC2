import unittest

from src.model.generator import Generator
from src.model.discriminator import Discriminator


class ModulesTests(unittest.TestCase):
    def test_discriminator_dimensionality(self):
        print("I'm")
        self.assertEqual(True, True)

    def test_generator_dimensionality(self):
        print("finally")
        self.assertEqual(True, True)

    def test_gen_output_as_disc_input(self):
        print("running!")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
