# 🔧 Troubleshooting GitHub Connection Issues

## Error: "Could not resolve hostname github.com"

Ini adalah masalah DNS/network, bukan masalah dengan repository Anda.

---

## ✅ Solusi Cepat (Pilih salah satu):

### **Solusi 1: Cek Koneksi Internet**
```powershell
# Test koneksi ke GitHub
ping github.com

# Jika gagal, coba:
ping 8.8.8.8  # Google DNS (test internet)
```

**Jika ping gagal:**
- ✅ Pastikan WiFi/Ethernet terhubung
- ✅ Restart router
- ✅ Coba browser untuk akses github.com

---

### **Solusi 2: Flush DNS Cache**
```powershell
# Clear DNS cache (Run as Administrator)
ipconfig /flushdns
ipconfig /registerdns

# Test lagi
ping github.com
```

---

### **Solusi 3: Ganti DNS ke Google/Cloudflare**

**Windows GUI:**
1. Control Panel → Network and Sharing Center
2. Change adapter settings → Right-click WiFi → Properties
3. IPv4 → Properties
4. Use the following DNS:
   - Preferred: `8.8.8.8` (Google)
   - Alternate: `8.8.4.4` (Google)
   - Atau: `1.1.1.1` (Cloudflare)

**PowerShell (Run as Admin):**
```powershell
# Set DNS to Google
Set-DnsClientServerAddress -InterfaceAlias "Wi-Fi" -ServerAddresses ("8.8.8.8","8.8.4.4")

# Atau Cloudflare
Set-DnsClientServerAddress -InterfaceAlias "Wi-Fi" -ServerAddresses ("1.1.1.1","1.0.0.1")

# Reset
ipconfig /flushdns
```

---

### **Solusi 4: Edit hosts File (Manual DNS)**

```powershell
# Buka Notepad as Administrator
notepad C:\Windows\System32\drivers\etc\hosts

# Tambahkan baris ini di akhir file:
140.82.121.4 github.com
140.82.121.3 gist.github.com
185.199.108.153 assets-cdn.github.com
199.232.69.194 github.global.ssl.fastly.net
```

**Save dan restart terminal**, lalu test:
```powershell
ping github.com
git ls-remote https://github.com/kevinnaufaldany/plain34-resnet34.git
```

---

### **Solusi 5: Gunakan GitHub CLI (Alternative)**

```powershell
# Install GitHub CLI
winget install --id GitHub.cli

# Login
gh auth login

# Push dengan gh
gh repo view
gh repo sync
```

---

### **Solusi 6: Gunakan HTTPS dengan Token (Bypass SSH)**

Jika SSH bermasalah, gunakan HTTPS:

```powershell
# Check remote URL
git remote -v

# Ganti ke HTTPS jika masih SSH
git remote set-url origin https://github.com/kevinnaufaldany/plain34-resnet34.git

# Push dengan HTTPS
git push origin main
```

**Untuk HTTPS, Anda perlu Personal Access Token (PAT):**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. Pilih scope: `repo` (full control)
4. Copy token
5. Saat push, gunakan token sebagai password

---

### **Solusi 7: Check Firewall/Antivirus**

```powershell
# Temporary disable Windows Firewall (untuk test)
# Control Panel → Windows Defender Firewall → Turn off

# Atau tambahkan exception untuk Git
netsh advfirewall firewall add rule name="Git" dir=out action=allow program="C:\Program Files\Git\bin\git.exe"
```

---

### **Solusi 8: VPN/Proxy Issues**

Jika Anda menggunakan VPN/Proxy:

```powershell
# Disable VPN sementara
# Atau configure Git proxy:

# Check current proxy
git config --global --get http.proxy

# Unset proxy
git config --global --unset http.proxy
git config --global --unset https.proxy

# Atau set proxy (jika kampus pakai proxy)
git config --global http.proxy http://proxy.server:port
git config --global https.proxy http://proxy.server:port
```

---

## 🧪 Diagnostic Commands

```powershell
# Test DNS resolution
nslookup github.com

# Test network connectivity
Test-NetConnection github.com -Port 443

# Check Git config
git config --list --show-origin

# Test Git connection
git ls-remote https://github.com/kevinnaufaldany/plain34-resnet34.git

# Check SSH (if using SSH)
ssh -T git@github.com
```

---

## 📋 Checklist Troubleshooting

- [ ] Internet connection working?
- [ ] Can access github.com in browser?
- [ ] DNS cache flushed?
- [ ] Firewall/Antivirus blocking?
- [ ] VPN/Proxy interfering?
- [ ] Using correct remote URL (HTTPS vs SSH)?
- [ ] GitHub PAT configured (for HTTPS)?
- [ ] SSH key configured (for SSH)?

---

## 🎯 Quick Fix untuk Push Sekarang

**Jika internet OK tapi Git masih error:**

```powershell
# 1. Ganti ke HTTPS
git remote set-url origin https://github.com/kevinnaufaldany/plain34-resnet34.git

# 2. Flush DNS
ipconfig /flushdns

# 3. Push
git push origin main

# 4. Masukkan username GitHub
# 5. Masukkan PAT (bukan password biasa!)
```

---

## 💡 Alternative: Save Work Locally First

Jika semua gagal, save dulu secara lokal:

```powershell
# Create backup
git bundle create backup.bundle --all

# Atau zip seluruh folder
Compress-Archive -Path . -DestinationPath ../per5_backup.zip

# Nanti saat internet OK, push lagi
git push origin main
```

---

## 📞 Masih Bermasalah?

1. **Test dengan GitHub Desktop** (GUI alternative)
2. **Test dengan hotspot HP** (bypass campus/home network)
3. **Test di jaringan lain** (cafe, perpustakaan, etc.)
4. **Contact IT support** (jika di kampus)

---

## ✅ Setelah Berhasil

```powershell
# Verify push
git log --oneline -n 5
git status

# Check remote
git remote -v
git ls-remote origin
```

---

**Good luck! 🚀**
