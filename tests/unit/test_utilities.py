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
