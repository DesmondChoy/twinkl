-- Render the editable source captions and tables with the original numbering.
local function has_class(element, name)
  return element.classes:includes(name)
end

local function caption_inlines(block)
  local content = block.content
  if #content == 1 and content[1].t == 'Strong' then
    content = content[1].content
  end
  local result = pandoc.List()
  for index = 5, #content do
    result:insert(content[index])
  end
  return result
end

-- Keep source straight double quotes from turning into curly glyphs.
function Str(element)
  if not element.text:find('"', 1, true) then return end
  local result = pandoc.List()
  local start = 1
  while true do
    local position = element.text:find('"', start, true)
    if not position then
      if start <= #element.text then result:insert(pandoc.Str(element.text:sub(start))) end
      break
    end
    if position > start then result:insert(pandoc.Str(element.text:sub(start, position - 1))) end
    result:insert(pandoc.RawInline('latex', '\\textquotedbl{}'))
    start = position + 1
  end
  return result
end

-- Let long visible URLs wrap while retaining their existing link destination.
function Code(element)
  if element.text:match('^https?://') then
    return pandoc.RawInline('latex', '\\path{' .. element.text .. '}')
  end
end

-- Preserve code verbatim while allowing long command lines to wrap.
function CodeBlock(element)
  return pandoc.RawBlock('latex', '\\begin{lstlisting}\n' .. element.text .. '\n\\end{lstlisting}')
end

function RawInline(element)
  if element.format == 'html' and element.text:match('^<br%s*/?>$') then
    return pandoc.LineBreak()
  end
end

function Div(element)
  if has_class(element, 'source-abstract') then
    local blocks = pandoc.List({pandoc.RawBlock('latex', '\\phantomsection\\addcontentsline{toc}{section}{Abstract}\\begin{abstract}')})
    for _, block in ipairs(element.content) do
      if block.t ~= 'Header' then blocks:insert(block) end
    end
    blocks:insert(pandoc.RawBlock('latex', '\\end{abstract}\\clearpage{\\small\\setlength{\\parskip}{0pt}\\tableofcontents}\\clearpage'))
    return blocks
  end

  if has_class(element, 'source-figure') then
    local content = pandoc.List()
    local panels = {}
    local caption
    for _, block in ipairs(element.content) do
      if block.t == 'Para' and pandoc.utils.stringify(block):match('^Figure %d+:') then
        caption = caption_inlines(block)
      elseif block.t == 'Div' and has_class(block, 'source-panel') then
        local panel = pandoc.write(pandoc.Pandoc(block.content), 'latex')
        panels[#panels + 1] = '\\begin{minipage}[t]{0.48\\linewidth}\\vspace{0pt}\\centering\\small\n' .. panel .. '\\end{minipage}'
      elseif block.t == 'Figure' then
        content:extend(block.content)
      else
        content:insert(block)
      end
    end
    assert(caption, 'Source figure is missing its caption: ' .. element.identifier)
    if #panels > 0 then
      content:insert(pandoc.RawBlock('latex', table.concat(panels, '\\hfill')))
    end
    local identifier = element.identifier:gsub('^source%-', '')
    caption:insert(pandoc.RawInline('latex', '\\label{' .. identifier .. '}'))
    local figure = pandoc.Figure(content, {pandoc.Plain(caption)}, pandoc.Attr(identifier))
    return {pandoc.RawBlock('latex', '\\FloatBarrier'), figure, pandoc.RawBlock('latex', '\\FloatBarrier')}
  end

  if has_class(element, 'source-table') then
    local caption
    local table_block
    for _, block in ipairs(element.content) do
      if block.t == 'Para' and pandoc.utils.stringify(block):match('^Table %d+:') then
        caption = caption_inlines(block)
      elseif block.t == 'Table' then
        table_block = block
      end
    end
    assert(table_block and caption, 'Source table is incomplete: ' .. element.identifier)
    table_block.identifier = element.identifier:gsub('^source%-', '')
    table_block.caption = {long = {pandoc.Plain(caption)}}
    local colspecs = table_block.colspecs
    local index = 1
    for width in element.attributes['column-widths']:gmatch('[^,]+') do
      colspecs[index][2] = tonumber(width)
      index = index + 1
    end
    table_block.colspecs = colspecs
    return {pandoc.RawBlock('latex', '\\FloatBarrier\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.18}'), table_block, pandoc.RawBlock('latex', '\\endgroup\\FloatBarrier')}
  end
end
