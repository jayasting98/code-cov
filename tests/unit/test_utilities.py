import os
import tempfile
import unittest

from code_cov import utilities


class UtilitiesTest(unittest.TestCase):
    def test_create_object_alias_decorator__class__creates_correctly(self):
        alias_classes: dict[str, type] = dict()
        class_alias = utilities.create_object_alias_decorator(alias_classes)
        @class_alias('class_alias')
        class ClassWithAlias:
            pass
        self.assertEqual(dict(class_alias=ClassWithAlias), alias_classes)

    def test_create_object_alias_decorator__function__creates_correctly(self):
        alias_functions: dict[str, type] = dict()
        function_alias = (
            utilities.create_object_alias_decorator(alias_functions))
        @function_alias('function_alias')
        def do_function_with_alias():
            pass
        self.assertEqual(
            dict(function_alias=do_function_with_alias), alias_functions)


class WorkingDirectoryTest(unittest.TestCase):
    def test___typical_case__executes_in_respective_working_directories(self):
        current_working_directory = os.getcwd()
        with utilities.WorkingDirectory('/tmp'):
            self.assertEqual('/tmp', os.getcwd())
        self.assertEqual(current_working_directory, os.getcwd())


class TemporaryChangeFileTest(unittest.TestCase):
    def test___typical_case__resets_file(self):
        with tempfile.TemporaryDirectory() as temp_dir_pathname:
            file_pathname = os.path.join(temp_dir_pathname, 'file.txt')
            with open(file_pathname, mode='w') as file:
                file.write('Hello World!\n')
            with open(file_pathname) as file:
                self.assertEqual('Hello World!\n', file.read())
            with utilities.TemporaryChangeFile(file_pathname):
                with open(file_pathname, mode='w') as file:
                    file.write('Goodbye World!\n')
                with open(file_pathname) as file:
                    self.assertEqual('Goodbye World!\n', file.read())
            with open(file_pathname) as file:
                self.assertEqual('Hello World!\n', file.read())
