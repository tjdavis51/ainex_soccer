import mujoco

m = mujoco.MjModel.from_xml_path("ainex/ainex.urdf")
xml = mujoco.mj_saveLastXML("ainex/ainex_exported.xml", m)
print("saved ainex/ainex_exported.xml")