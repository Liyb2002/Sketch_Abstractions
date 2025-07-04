class Component:
    def __init__(self, name, parameters, location, cad_operations, quantity=1, locations=None, children=None):
        """
        Represents a component of a macro.

        :param name: Name of the component
        :param parameters: Dict of parameter ranges
        :param location: Default location (dict)
        :param cad_operations: List of CAD operations (e.g., ["sketch_rectangle", "extrude"])
        :param quantity: How many of this component
        :param locations: Optional list of specific locations (length = quantity)
        :param children: Optional list of child Component instances
        """
        self.name = name
        self.parameters = parameters
        self.location = location
        self.cad_operations = cad_operations
        self.quantity = quantity
        self.locations = locations if locations else []
        self.children = children if children else []

    def __repr__(self):
        return (
            f"Component(name={self.name!r}, quantity={self.quantity}, "
            f"cad_operations={self.cad_operations}, children={len(self.children)})"
        )

    def describe(self, indent=0):
        pad = "  " * indent
        loc_info = (
            f", locations={len(self.locations)}"
            if self.quantity > 1 and self.locations
            else ""
        )
        desc = (
            f"{pad}- {self.name} (x{self.quantity}{loc_info}): "
            f"{' → '.join(self.cad_operations)}"
        )
        print(desc)
        for child in self.children:
            child.describe(indent + 1)

    def build(self):
        """
        Placeholder: build the shape using CAD library.
        To be implemented later.
        """
        raise NotImplementedError("Build method not yet implemented.")
