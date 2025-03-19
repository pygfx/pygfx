"""
Big static text
===============

Render 10 paragraphs of Lorem ipsum.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from rendercanvas.auto import RenderCanvas, loop

import pygfx as gfx


canvas = RenderCanvas(
    size=(800, 600), title="Big static text $fps", update_mode="continuous"
)
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()


scene.add(gfx.Background.from_color("#fff"))

text = """
Lorem ipsum odor amet, consectetuer adipiscing elit. Congue aliquet fusce hendrerit leo fames ac. Proin nec sit mauris lobortis quam ultrices. Senectus habitasse ad orci posuere fusce. Ut lectus inceptos commodo taciti porttitor a habitasse. Vulputate tempus ullamcorper aptent molestie vestibulum massa. Tristique nec ac sagittis morbi; egestas nisl donec morbi. Et nisi donec conubia duis rutrum tellus?

Amet praesent proin leo morbi dignissim eu et sagittis. Finibus dictum parturient eleifend in interdum class mauris. Nisi mollis est per; aliquet primis interdum. Semper torquent sem aliquam eleifend id malesuada. Magnis tincidunt suspendisse sapien massa etiam aptent libero morbi. Commodo montes ac torquent quam rutrum. Ante dis potenti nostra maximus non mattis. Interdum fringilla suspendisse parturient accumsan habitant ridiculus proin. Cursus dolor facilisis arcu ultricies facilisi velit habitasse natoque.

Netus libero nascetur senectus habitant viverra cras euismod maximus. Mattis praesent ornare morbi donec commodo. Natoque ante taciti primis habitasse malesuada montes euismod. Quisque id faucibus, condimentum vehicula montes fusce sem. Eu congue malesuada justo a nulla suscipit. Aplatea curabitur et eget felis, at aptent velit. Ultricies id viverra vestibulum eleifend, massa sed nec. Magnis nullam curae urna quam condimentum facilisis eu. Euismod euismod phasellus, tempus inceptos risus accumsan morbi. Curae metus libero volutpat nisi quis facilisi montes.

Netus egestas magnis cras habitasse porta egestas fringilla. Tortor lectus tortor iaculis eu arcu ex sollicitudin quis. Nascetur dis malesuada ad dapibus, lobortis maximus leo inceptos eros. Vel auctor libero iaculis aliquet fringilla leo donec; curabitur eleifend. Quis laoreet placerat cursus dui sapien finibus. Mi gravida dapibus ante libero tristique auctor lobortis. Primis litora vivamus enim blandit sed.

Nisl eleifend ante mus fusce; feugiat habitasse montes? Volutpat convallis ullamcorper viverra quisque neque porttitor sit? Tempus ut ligula ex maximus ultricies justo erat tempus. Risus bibendum malesuada vitae nulla erat fermentum luctus. Viverra enim nisi cursus sodales tortor volutpat eu? Porta libero mauris ligula elementum facilisi fermentum dignissim. Himenaeos convallis nunc viverra ligula litora mauris tempor etiam? Sem accumsan nibh adipiscing proin sagittis hendrerit suscipit dignissim.

Sapien fringilla facilisis nisl neque leo justo natoque quam. Integer vehicula suspendisse vehicula metus hac rhoncus sed. Lectus felis torquent suspendisse ullamcorper sem est tellus habitasse cras. Dictumst justo nunc vitae quisque facilisis. Vitae bibendum habitant semper nullam purus tellus phasellus faucibus. Arcu nascetur penatibus efficitur ridiculus nullam elit. Mauris nostra ridiculus nibh bibendum tempus sollicitudin ipsum non eleifend.

Ultrices himenaeos litora cubilia risus magna hac. Nisi donec sit praesent pellentesque molestie taciti dui. Laoreet tristique odio natoque elementum duis platea posuere. Dolor adipiscing ligula posuere sodales cubilia, taciti condimentum dignissim. Viverra eget class torquent fusce condimentum. Conubia laoreet viverra dictumst habitant, natoque potenti senectus. Sem leo diam suscipit massa faucibus congue. Faucibus cras ad leo pretium augue iaculis ex mus. Quisque malesuada torquent elit curabitur mauris eleifend? Viverra interdum enim dictum curae fringilla potenti felis dolor duis.

Vitae maecenas varius justo egestas eleifend finibus congue. Orci eu taciti sagittis mus nibh congue urna conubia. Habitant duis nec vulputate orci congue quam. Est sem amet nulla blandit eget. Litora ut est lobortis natoque commodo; eros molestie rhoncus. Inceptos commodo lectus ante porttitor primis duis libero inceptos. Ipsum convallis libero torquent mattis etiam sed finibus. Hac leo ligula; dis fames molestie elit. Cras ridiculus euismod sollicitudin sodales nec ante fusce tristique.

Elit curabitur condimentum eros faucibus rhoncus faucibus taciti. Congue enim consequat venenatis torquent ante senectus neque tortor tincidunt. Odio litora nisl pharetra magnis arcu lacus. Vulputate amet velit nascetur arcu sociosqu erat scelerisque. Habitant ipsum dis faucibus eros tempor tellus. Eleifend class ex non consequat nunc conubia et hendrerit ligula. Porttitor vulputate dolor phasellus efficitur nisi viverra. Orci ridiculus senectus sit nostra conubia purus sociosqu mauris. Elementum nisl tristique magnis vulputate enim efficitur. Dignissim phasellus integer quisque dapibus sollicitudin conubia montes risus.

Eleifend imperdiet primis quam conubia sapien lacus nec sagittis curabitur. Dapibus ante laoreet integer euismod sit ad nulla ultricies. Aenean neque libero lacus ut mollis mus? Cubilia nascetur tempus ultrices vel cursus ipsum egestas vivamus varius. Tellus nostra tristique quis ante felis. Per maximus odio eget nisl ornare; faucibus risus natoque. Curae dapibus interdum finibus adipiscing dui duis convallis eleifend. Ligula ornare himenaeos dictumst felis nam tempor tempor. Curae penatibus tortor vel adipiscing quam nec penatibus himenaeos.
"""


tob = gfx.Text(
    text=text,
    max_width=1000,
    material=gfx.TextMaterial(color="#000"),
)
scene.add(tob)

camera = gfx.OrthographicCamera(900, 900)
controller = gfx.PanZoomController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)


renderer.request_draw(animate)


if __name__ == "__main__":
    loop.run()
